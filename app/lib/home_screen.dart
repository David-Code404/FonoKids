import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'api_client.dart';
import 'app_theme.dart';
import 'mock_data.dart';
import 'settings_dialog.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

enum _ConnState { unknown, ok, fail }

/// Debajo de esto, la predicción no cuenta como "frase de riesgo" -- se
/// muestra un instante en el panel de capturas y se descarta solo.
const double _riskThreshold = 0.40;

class _CaptureEntry {
  final int id;
  final String word;
  final double prob;
  final bool isRisk;
  final DateTime time;

  _CaptureEntry({
    required this.id,
    required this.word,
    required this.prob,
    required this.isRisk,
    required this.time,
  });
}

const _avatarPalette = [
  AppColors.accent,
  AppColors.accent2,
  AppColors.accent3,
  Color(0xFFFF6B81),
  Color(0xFF5AA9FF),
];

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  Future<void>? _initFuture;
  String? _cameraError;

  List<CameraDescription> _cameras = [];
  int _selectedCameraIndex = 0;
  bool _cameraPicked = false;

  bool _recording = false;
  bool _predicting = false;
  String _serverUrl = ApiClient.defaultUrl;
  _ConnState _connState = _ConnState.unknown;

  PredictionResult? _lastResult;
  String? _lastError;

  final List<_CaptureEntry> _captures = [];
  final Set<int> _fadingOutIds = {};
  int _nextCaptureId = 0;
  int _discardedCount = 0;
  final List<Timer> _pendingTimers = [];

  // Maqueta de "una vista por persona detectada" (todavía no hay detección
  // de varias caras real -- ver nota en el código de _buildPersonViews).
  int _mockPersonCount = 2;
  int? _expandedPersonIndex;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    if (MockData.previewOnly) _seedPreviewCaptures();
    _bootstrap();
  }

  /// Un par de capturas de muestra para poder ver el panel funcionando sin
  /// depender de la cámara ni del servidor.
  void _seedPreviewCaptures() {
    _registerCapture('te_voy_a_matar', 0.87);
    _registerCapture('hola_buenas_tardes', 0.22);
  }

  Future<void> _bootstrap() async {
    _serverUrl = await ApiClient.getServerUrl();
    await _initCamera();
    unawaited(_refreshConnection());
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _cameraError = 'Este dispositivo no tiene cámaras disponibles.');
        return;
      }
      _cameras = cameras;
      if (_selectedCameraIndex >= cameras.length) _selectedCameraIndex = 0;
      if (!_cameraPicked) {
        // La primera vez, preferir la cámara frontal si hay más de una.
        final frontIdx = cameras.indexWhere((c) => c.lensDirection == CameraLensDirection.front);
        _selectedCameraIndex = frontIdx != -1 ? frontIdx : 0;
        _cameraPicked = true;
      }
      final selected = cameras[_selectedCameraIndex];

      final controller = CameraController(
        selected,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      _controller = controller;
      _initFuture = controller.initialize();
      await _initFuture;
      if (!mounted) return;
      setState(() => _cameraError = null);
    } catch (e) {
      setState(() => _cameraError = 'No se pudo iniciar la cámara: $e');
    }
  }

  /// Cambia a otra cámara conectada (webcam externa, otra lente, etc.) o se
  /// puede quedar con una sola si es la única disponible.
  Future<void> _switchCamera(int index) async {
    if (index == _selectedCameraIndex || index >= _cameras.length) return;
    final oldController = _controller;
    setState(() {
      _selectedCameraIndex = index;
      _cameraError = null;
      _controller = null;
    });
    await oldController?.dispose();
    await _initCamera();
  }

  /// Vuelve a escanear las cámaras conectadas a la PC (útil si enchufaste
  /// una webcam nueva o prendiste una cámara virtual después de abrir la
  /// app) sin tener que reiniciar nada.
  Future<List<CameraDescription>> _rescanCameras() async {
    final cams = await availableCameras();
    if (mounted) setState(() => _cameras = cams);
    return cams;
  }

  String _cameraLabel(CameraDescription cam) {
    final lens = switch (cam.lensDirection) {
      CameraLensDirection.front => 'frontal',
      CameraLensDirection.back => 'trasera',
      CameraLensDirection.external => 'externa',
    };
    return cam.name.isNotEmpty ? '${cam.name} ($lens)' : 'Cámara $lens';
  }

  Future<void> _refreshConnection() async {
    final ok = await ApiClient.checkHealth(_serverUrl);
    if (!mounted) return;
    setState(() => _connState = ok ? _ConnState.ok : _ConnState.fail);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;

    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      controller.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    for (final t in _pendingTimers) {
      t.cancel();
    }
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _openSettings() async {
    final updated = await showDialog<String>(
      context: context,
      builder: (_) => SettingsDialog(currentUrl: _serverUrl),
    );
    if (updated != null && updated.isNotEmpty) {
      setState(() {
        _serverUrl = updated;
        _connState = _ConnState.unknown;
      });
      await ApiClient.setServerUrl(updated);
      unawaited(_refreshConnection());
    }
  }

  /// Agrega una predicción al panel de "personas capturadas". Si es una
  /// frase de riesgo entrenada, se queda ahí (después pasaría a formar
  /// parte del historial permanente de Capturas). Si no es una frase de
  /// riesgo (o la confianza es muy baja), se muestra un momento y se
  /// elimina sola -- no vale la pena guardarla.
  void _registerCapture(String word, double prob) {
    final entry = _CaptureEntry(
      id: _nextCaptureId++,
      word: word,
      prob: prob,
      isRisk: prob >= _riskThreshold,
      time: DateTime.now(),
    );
    setState(() => _captures.insert(0, entry));

    if (!entry.isRisk) {
      final fadeTimer = Timer(const Duration(seconds: 5), () {
        if (!mounted) return;
        setState(() => _fadingOutIds.add(entry.id));
      });
      final removeTimer = Timer(const Duration(milliseconds: 5600), () {
        if (!mounted) return;
        setState(() {
          _captures.removeWhere((c) => c.id == entry.id);
          _fadingOutIds.remove(entry.id);
          _discardedCount++;
        });
      });
      _pendingTimers.addAll([fadeTimer, removeTimer]);
    }
  }

  void _dismissCapture(int id) {
    setState(() {
      _captures.removeWhere((c) => c.id == id);
      _fadingOutIds.remove(id);
    });
  }

  void _clearCaptures() {
    setState(() {
      _captures.clear();
      _fadingOutIds.clear();
    });
  }

  Future<void> _toggleRecording() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;

    if (!_recording) {
      try {
        setState(() {
          _lastResult = null;
          _lastError = null;
        });
        await controller.startVideoRecording();
        setState(() => _recording = true);
      } catch (e) {
        setState(() => _lastError = 'No se pudo empezar a grabar: $e');
      }
      return;
    }

    setState(() {
      _recording = false;
      _predicting = true;
    });

    try {
      final file = await controller.stopVideoRecording();
      final result = await ApiClient.predict(_serverUrl, File(file.path));
      if (!mounted) return;
      setState(() {
        _lastResult = result;
        _connState = _ConnState.ok;
      });
      if (result.valid && result.word != null) {
        _registerCapture(result.word!, result.prob ?? 0);
      }
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() {
        _lastError = e.message;
        _connState = _ConnState.fail;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _lastError = 'Error inesperado: $e');
    } finally {
      if (mounted) setState(() => _predicting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final showSidePanel = constraints.maxWidth >= 760;
            final cameraArea = Stack(
              fit: StackFit.expand,
              children: [
                MockData.previewOnly ? _buildPersonViews() : _buildCameraPreview(),
                _buildTopBar(),
                if (MockData.previewOnly && _expandedPersonIndex == null) _personCountControl(),
                if (_expandedPersonIndex == null) _buildStatusArea(),
                if (_expandedPersonIndex == null) _buildRecordControl(),
              ],
            );

            if (!showSidePanel) return cameraArea;

            return Row(
              children: [
                Expanded(child: cameraArea),
                _CapturesPanel(
                  captures: _captures,
                  fadingOutIds: _fadingOutIds,
                  discardedCount: _discardedCount,
                  onDismiss: _dismissCapture,
                  onClear: _captures.isEmpty ? null : _clearCaptures,
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  /// Maqueta de la vista "una cámara por persona": si hay 1 persona en
  /// cuadro se ve 1 recuadro, si hay 2 se ven 2, etc. TODAVÍA es una
  /// simulación con datos de muestra -- para que fuera real, la app
  /// necesitaría detectar varias caras en vivo (hoy no lo hace, solo manda
  /// el video entero al servidor).
  Widget _buildPersonViews() {
    if (_expandedPersonIndex != null) {
      final i = _expandedPersonIndex!;
      return _PersonTile(
        index: i,
        fullscreen: true,
        onTap: () => setState(() => _expandedPersonIndex = null),
      );
    }

    return Container(
      color: Colors.black,
      padding: const EdgeInsets.fromLTRB(16, 64, 16, 130),
      child: GridView.builder(
        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: _columnsFor(_mockPersonCount),
          mainAxisSpacing: 10,
          crossAxisSpacing: 10,
          childAspectRatio: 1.6,
        ),
        itemCount: _mockPersonCount,
        itemBuilder: (context, i) => _PersonTile(
          index: i,
          fullscreen: false,
          onTap: () => setState(() => _expandedPersonIndex = i),
        ),
      ),
    );
  }

  /// Cuántas columnas usar según cuántas personas hay, para que no quede
  /// una fila suelta gigante (ej. 3 personas en 3 columnas, no 2+1).
  int _columnsFor(int n) {
    if (n <= 1) return 1;
    if (n == 3) return 3;
    if (n <= 4) return 2;
    return 3;
  }

  Widget _personCountControl() {
    const options = [1, 2, 3, 4, 6];
    return Positioned(
      top: 54,
      left: 8,
      right: 8,
      child: Center(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: Colors.black.withValues(alpha: 0.55),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: AppColors.border),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Padding(
                padding: EdgeInsets.only(right: 8),
                child: Text(
                  'Simular personas en cuadro:',
                  style: TextStyle(color: Colors.white70, fontSize: 11.5),
                ),
              ),
              for (final n in options)
                GestureDetector(
                  onTap: () => setState(() => _mockPersonCount = n),
                  child: Container(
                    margin: const EdgeInsets.symmetric(horizontal: 2),
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                    decoration: BoxDecoration(
                      color: _mockPersonCount == n ? AppColors.accent : Colors.transparent,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      '$n',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: _mockPersonCount == n ? FontWeight.w800 : FontWeight.w500,
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (_cameraError != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            _cameraError!,
            style: const TextStyle(color: Colors.white70, fontSize: 16),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    final controller = _controller;
    if (controller == null || _initFuture == null) {
      return const Center(child: CircularProgressIndicator());
    }

    return FutureBuilder<void>(
      future: _initFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }
        return Center(child: CameraPreview(controller));
      },
    );
  }

  Widget _buildTopBar() {
    return Positioned(
      top: 8,
      left: 8,
      right: 8,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          _connectionChip(),
          Row(
            children: [
              _cameraSwitchButton(),
              IconButton(
                icon: const Icon(Icons.settings, color: Colors.white, size: 28),
                onPressed: _openSettings,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _cameraSwitchButton() {
    return IconButton(
      tooltip: 'Cámaras',
      icon: const Icon(Icons.cameraswitch_rounded, color: Colors.white, size: 25),
      onPressed: _openCameraPicker,
    );
  }

  /// Abre la hoja para elegir cámara. Escanea de una así, si conectaste
  /// algo nuevo justo antes de tocar el botón, ya aparece.
  Future<void> _openCameraPicker() async {
    await _rescanCameras();
    if (!mounted) return;
    await showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(22)),
      ),
      builder: (sheetContext) {
        return StatefulBuilder(
          builder: (sheetContext, setSheetState) {
            return SafeArea(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 18, 20, 24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          'Cámaras disponibles',
                          style: TextStyle(
                            color: AppColors.textPrimary,
                            fontSize: 17,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        TextButton.icon(
                          onPressed: () async {
                            await _rescanCameras();
                            setSheetState(() {});
                          },
                          icon: const Icon(Icons.refresh_rounded, size: 18),
                          label: const Text('Buscar de nuevo'),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    const Text(
                      'Conectá una webcam USB o prendé una cámara virtual (ej. OBS) y '
                      'tocá "Buscar de nuevo" para que aparezca acá.',
                      style: TextStyle(color: AppColors.textSecondary, fontSize: 12, height: 1.4),
                    ),
                    const SizedBox(height: 14),
                    if (_cameras.isEmpty)
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 18),
                        child: Text(
                          'No se encontró ninguna cámara.',
                          style: TextStyle(color: AppColors.textMuted, fontSize: 13),
                        ),
                      )
                    else
                      for (var i = 0; i < _cameras.length; i++)
                        _cameraTile(sheetContext, i),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  Widget _cameraTile(BuildContext sheetContext, int i) {
    final selected = i == _selectedCameraIndex;
    return InkWell(
      borderRadius: BorderRadius.circular(14),
      onTap: () {
        Navigator.of(sheetContext).pop();
        _switchCamera(i);
      },
      child: Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: selected ? AppColors.accent.withValues(alpha: 0.12) : AppColors.surfaceAlt,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: selected ? AppColors.accent : AppColors.border),
        ),
        child: Row(
          children: [
            Icon(
              selected ? Icons.radio_button_checked_rounded : Icons.radio_button_off_rounded,
              size: 19,
              color: selected ? AppColors.accent : AppColors.textMuted,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                _cameraLabel(_cameras[i]),
                style: const TextStyle(color: AppColors.textPrimary, fontSize: 13.5),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _connectionChip() {
    Color color;
    String label;
    switch (_connState) {
      case _ConnState.ok:
        color = AppColors.accent2;
        label = 'Servidor OK';
        break;
      case _ConnState.fail:
        color = AppColors.danger;
        label = 'Sin conexión';
        break;
      case _ConnState.unknown:
        color = AppColors.textMuted;
        label = 'Conectando...';
        break;
    }
    return GestureDetector(
      onTap: _refreshConnection,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.55),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: AppColors.border),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            ),
            const SizedBox(width: 8),
            Text(label,
                style: const TextStyle(
                    color: Colors.white, fontSize: 12.5, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }

  /// Tarjetas de error/resultado -- van arriba, lejos del centro, para no
  /// taparte la vista de las personas.
  Widget _buildStatusArea() {
    if (_lastError == null && _lastResult == null) return const SizedBox.shrink();
    return Positioned(
      top: 90,
      left: 0,
      right: 0,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_lastError != null) _errorCard(_lastError!),
          if (_lastResult != null) _resultCard(_lastResult!),
        ],
      ),
    );
  }

  /// Control de grabar, en una esquina chica para no estorbar la vista.
  Widget _buildRecordControl() {
    return Positioned(
      right: 20,
      bottom: 20,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Container(
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.55),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              _predicting
                  ? 'Analizando...'
                  : (_recording ? 'Grabando... tocá para terminar' : 'Tocá para grabar'),
              style: const TextStyle(color: Colors.white70, fontSize: 11.5),
            ),
          ),
          _recordButton(),
        ],
      ),
    );
  }

  Widget _errorCard(String message) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 6),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.danger.withValues(alpha: 0.94),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        children: [
          const Icon(Icons.warning_amber_rounded, color: Colors.white),
          const SizedBox(width: 10),
          Expanded(
            child: Text(message, style: const TextStyle(color: Colors.white, fontSize: 14)),
          ),
        ],
      ),
    );
  }

  Widget _resultCard(PredictionResult result) {
    if (!result.valid) {
      return _errorCard(result.reason ?? 'Toma no válida, repetí.');
    }

    final isRisk = (result.prob ?? 0) >= _riskThreshold;
    final pct = ((result.prob ?? 0) * 100).toStringAsFixed(1);
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 6),
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
        boxShadow: const [BoxShadow(color: Colors.black45, blurRadius: 16)],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            result.word?.replaceAll('_', ' ') ?? '?',
            style: const TextStyle(
              fontSize: 26,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
              letterSpacing: -0.3,
            ),
          ),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
            decoration: BoxDecoration(
              color: (isRisk ? AppColors.danger : AppColors.accent2).withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              isRisk ? 'Frase de riesgo -- $pct%' : 'No es una frase de riesgo -- $pct%',
              style: TextStyle(
                fontSize: 12.5,
                color: isRisk ? AppColors.danger : AppColors.accent2,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _recordButton() {
    final busy = _predicting;
    return GestureDetector(
      onTap: busy ? null : _toggleRecording,
      child: Container(
        width: 60,
        height: 60,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: Colors.black.withValues(alpha: 0.4),
          border: Border.all(
            color: _recording ? AppColors.danger : Colors.white,
            width: 3.5,
          ),
        ),
        padding: const EdgeInsets.all(5),
        child: busy
            ? const Padding(
                padding: EdgeInsets.all(10),
                child: CircularProgressIndicator(color: Colors.white, strokeWidth: 3),
              )
            : AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                decoration: BoxDecoration(
                  shape: _recording ? BoxShape.rectangle : BoxShape.circle,
                  borderRadius: _recording ? BorderRadius.circular(8) : null,
                  color: AppColors.danger,
                ),
              ),
      ),
    );
  }
}

/// Panel lateral derecho con las personas capturadas en esta sesión. Las
/// frases de riesgo quedan fijas (rojo); las que no son de riesgo se
/// muestran un momento y se apagan solas.
class _CapturesPanel extends StatelessWidget {
  final List<_CaptureEntry> captures;
  final Set<int> fadingOutIds;
  final int discardedCount;
  final ValueChanged<int> onDismiss;
  final VoidCallback? onClear;

  const _CapturesPanel({
    required this.captures,
    required this.fadingOutIds,
    required this.discardedCount,
    required this.onDismiss,
    required this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    final riskCount = captures.where((c) => c.isRisk).length;

    return Container(
      width: 340,
      decoration: BoxDecoration(
        color: AppColors.surface,
        border: const Border(left: BorderSide(color: AppColors.border)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 20, 18, 12),
            child: Row(
              children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    gradient: AppColors.heroGradient,
                    borderRadius: BorderRadius.circular(11),
                  ),
                  child: const Icon(Icons.groups_rounded, color: Colors.white, size: 18),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Personas capturadas',
                        style: TextStyle(
                          color: AppColors.textPrimary,
                          fontSize: 14.5,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      Text(
                        '${captures.length} en esta sesión',
                        style: const TextStyle(color: AppColors.textMuted, fontSize: 11),
                      ),
                    ],
                  ),
                ),
                if (onClear != null)
                  IconButton(
                    icon: const Icon(Icons.delete_sweep_rounded, size: 19),
                    color: AppColors.textMuted,
                    tooltip: 'Limpiar sesión',
                    onPressed: onClear,
                  ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 0, 18, 12),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.info_outline_rounded, size: 13, color: AppColors.textMuted),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    'Cada detección es independiente: todavía no reconoce si dos '
                    'capturas son la misma persona.',
                    style: TextStyle(color: AppColors.textMuted.withValues(alpha: 0.9), fontSize: 10.5, height: 1.3),
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 0, 18, 14),
            child: Row(
              children: [
                Expanded(
                  child: _miniStat(
                    icon: Icons.warning_rounded,
                    color: AppColors.danger,
                    value: '$riskCount',
                    label: 'De riesgo',
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _miniStat(
                    icon: Icons.timer_outlined,
                    color: AppColors.textMuted,
                    value: '$discardedCount',
                    label: 'Descartadas',
                  ),
                ),
              ],
            ),
          ),
          const Divider(color: AppColors.border, height: 1),
          Expanded(
            child: captures.isEmpty
                ? const Center(
                    child: Padding(
                      padding: EdgeInsets.all(24),
                      child: Text(
                        'Todavía no se detectó a nadie.\nCuando alguien hable, va a aparecer acá.',
                        textAlign: TextAlign.center,
                        style: TextStyle(color: AppColors.textMuted, fontSize: 12.5, height: 1.4),
                      ),
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.fromLTRB(14, 14, 14, 16),
                    itemCount: captures.length,
                    itemBuilder: (context, i) {
                      final entry = captures[i];
                      return AnimatedOpacity(
                        duration: const Duration(milliseconds: 600),
                        opacity: fadingOutIds.contains(entry.id) ? 0.0 : 1.0,
                        child: _CaptureListCard(
                          entry: entry,
                          onDismiss: () => onDismiss(entry.id),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _miniStat({
    required IconData icon,
    required Color color,
    required String value,
    required String label,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha: 0.25)),
      ),
      child: Row(
        children: [
          Icon(icon, size: 15, color: color),
          const SizedBox(width: 7),
          Text(
            value,
            style: TextStyle(color: color, fontSize: 14, fontWeight: FontWeight.w800),
          ),
          const SizedBox(width: 5),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(color: AppColors.textMuted, fontSize: 10.5),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}

class _CaptureListCard extends StatelessWidget {
  final _CaptureEntry entry;
  final VoidCallback onDismiss;

  const _CaptureListCard({required this.entry, required this.onDismiss});

  Color get _avatarColor => _avatarPalette[entry.id % _avatarPalette.length];

  String get _timeLabel {
    final t = entry.time;
    final hh = t.hour.toString().padLeft(2, '0');
    final mm = t.minute.toString().padLeft(2, '0');
    return '$hh:$mm';
  }

  @override
  Widget build(BuildContext context) {
    final color = _avatarColor;
    return GestureDetector(
      onTap: () => _showEnlarged(context, color),
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: AppColors.surfaceAlt,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: entry.isRisk ? AppColors.danger.withValues(alpha: 0.4) : AppColors.border,
          ),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 60,
              height: 60,
              alignment: Alignment.center,
              clipBehavior: Clip.antiAlias,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [color.withValues(alpha: 0.55), color.withValues(alpha: 0.18)],
                ),
              ),
              child: Icon(Icons.person_rounded, color: Colors.white.withValues(alpha: 0.85), size: 32),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          entry.word.replaceAll('_', ' '),
                          style: const TextStyle(
                            color: AppColors.textPrimary,
                            fontSize: 13.5,
                            fontWeight: FontWeight.w700,
                          ),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      GestureDetector(
                        onTap: onDismiss,
                        child: const Padding(
                          padding: EdgeInsets.only(left: 6),
                          child: Icon(Icons.close_rounded, size: 16, color: AppColors.textMuted),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 3),
                  const Text(
                    'Persona no identificada',
                    style: TextStyle(
                      color: AppColors.textMuted,
                      fontSize: 10,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                  const SizedBox(height: 3),
                  Text(
                    '$_timeLabel · ${(entry.prob * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(color: AppColors.textMuted, fontSize: 10.5),
                  ),
                  const SizedBox(height: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: (entry.isRisk ? AppColors.danger : AppColors.textMuted)
                          .withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          entry.isRisk ? Icons.warning_rounded : Icons.timer_outlined,
                          size: 11,
                          color: entry.isRisk ? AppColors.danger : AppColors.textMuted,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          entry.isRisk ? 'Guardado en Capturas' : 'Descartando...',
                          style: TextStyle(
                            color: entry.isRisk ? AppColors.danger : AppColors.textMuted,
                            fontSize: 9.5,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showEnlarged(BuildContext context, Color color) {
    showDialog(
      context: context,
      builder: (_) => Dialog(
        backgroundColor: Colors.transparent,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(22),
          child: Container(
            width: 300,
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(22),
              border: Border.all(color: AppColors.border),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Stack(
                  children: [
                    AspectRatio(
                      aspectRatio: 1,
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [color.withValues(alpha: 0.55), color.withValues(alpha: 0.18)],
                          ),
                        ),
                        child: Stack(
                          alignment: Alignment.center,
                          children: [
                            Icon(Icons.person_rounded, size: 110, color: Colors.white.withValues(alpha: 0.85)),
                            Positioned(
                              bottom: 10,
                              left: 10,
                              right: 10,
                              child: Container(
                                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
                                decoration: BoxDecoration(
                                  color: Colors.black.withValues(alpha: 0.45),
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                child: const Text(
                                  'Imagen de muestra -- todavía no hay foto real conectada',
                                  style: TextStyle(color: Colors.white70, fontSize: 10),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    Positioned(
                      top: 6,
                      right: 6,
                      child: IconButton(
                        icon: const Icon(Icons.close_rounded, color: Colors.white),
                        onPressed: () => Navigator.of(context).pop(),
                      ),
                    ),
                  ],
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(18, 14, 18, 18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        entry.word.replaceAll('_', ' '),
                        style: const TextStyle(
                          color: AppColors.textPrimary,
                          fontSize: 18,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Detectado a las $_timeLabel con ${(entry.prob * 100).toStringAsFixed(0)}% de confianza.',
                        style: const TextStyle(color: AppColors.textSecondary, fontSize: 12.5),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// Un recuadro de la maqueta "una vista por persona". `fullscreen` es la
/// versión ampliada al tocar el recuadro.
class _PersonTile extends StatelessWidget {
  final int index;
  final bool fullscreen;
  final VoidCallback onTap;

  const _PersonTile({required this.index, required this.fullscreen, required this.onTap});

  Color get _color => _avatarPalette[index % _avatarPalette.length];

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(fullscreen ? 0 : 16),
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [_color.withValues(alpha: 0.5), _color.withValues(alpha: 0.15)],
          ),
          border: fullscreen ? null : Border.all(color: AppColors.border),
        ),
        child: Stack(
          alignment: Alignment.center,
          children: [
            Icon(
              Icons.person_rounded,
              size: fullscreen ? 160 : 46,
              color: Colors.white.withValues(alpha: 0.85),
            ),
            Positioned(
              left: 10,
              top: 10,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.45),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Persona ${index + 1}',
                  style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w700),
                ),
              ),
            ),
            Positioned(
              right: 10,
              top: 10,
              child: Icon(
                fullscreen ? Icons.close_fullscreen_rounded : Icons.open_in_full_rounded,
                color: Colors.white70,
                size: 16,
              ),
            ),
            Positioned(
              bottom: 10,
              child: Text(
                fullscreen ? 'Vista simulada -- tocá para volver' : 'Vista simulada',
                style: const TextStyle(color: Colors.white60, fontSize: 10),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
