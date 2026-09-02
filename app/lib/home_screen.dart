import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'api_client.dart';
import 'dataset_dashboard_screen.dart';
import 'settings_dialog.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

enum _ConnState { unknown, ok, fail }

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  Future<void>? _initFuture;
  String? _cameraError;

  bool _recording = false;
  bool _predicting = false;
  String _serverUrl = ApiClient.defaultUrl;
  _ConnState _connState = _ConnState.unknown;

  PredictionResult? _lastResult;
  String? _lastError;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _bootstrap();
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
      final selected = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

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
        child: Stack(
          fit: StackFit.expand,
          children: [
            _buildCameraPreview(),
            _buildTopBar(),
            _buildBottomPanel(),
          ],
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
              IconButton(
                icon: const Icon(Icons.bar_chart, color: Colors.white, size: 28),
                tooltip: 'Progreso del dataset',
                onPressed: _openDashboard,
              ),
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

  void _openDashboard() {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => DatasetDashboardScreen(serverUrl: _serverUrl),
      ),
    );
  }

  Widget _connectionChip() {
    Color color;
    String label;
    switch (_connState) {
      case _ConnState.ok:
        color = Colors.greenAccent;
        label = 'Servidor OK';
        break;
      case _ConnState.fail:
        color = Colors.redAccent;
        label = 'Sin conexión';
        break;
      case _ConnState.unknown:
        color = Colors.grey;
        label = 'Conectando...';
        break;
    }
    return GestureDetector(
      onTap: _refreshConnection,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 10,
              height: 10,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            ),
            const SizedBox(width: 8),
            Text(label, style: const TextStyle(color: Colors.white, fontSize: 13)),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomPanel() {
    return Positioned(
      left: 0,
      right: 0,
      bottom: 24,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_lastError != null) _errorCard(_lastError!),
          if (_lastResult != null) _resultCard(_lastResult!),
          const SizedBox(height: 16),
          _recordButton(),
          const SizedBox(height: 8),
          Text(
            _predicting
                ? 'Analizando...'
                : (_recording ? 'Grabando... tocá para terminar' : 'Tocá para grabar'),
            style: const TextStyle(color: Colors.white70, fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _errorCard(String message) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 6),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.orange.shade800.withValues(alpha: 0.92),
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

    final pct = ((result.prob ?? 0) * 100).toStringAsFixed(1);
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 6),
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.95),
        borderRadius: BorderRadius.circular(18),
        boxShadow: const [BoxShadow(color: Colors.black26, blurRadius: 12)],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            result.word ?? '?',
            style: const TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
              color: Colors.black87,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Confianza: $pct%',
            style: const TextStyle(fontSize: 14, color: Colors.black54),
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
        width: 78,
        height: 78,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white, width: 4),
          color: Colors.transparent,
        ),
        padding: const EdgeInsets.all(6),
        child: busy
            ? const Padding(
                padding: EdgeInsets.all(14),
                child: CircularProgressIndicator(color: Colors.white, strokeWidth: 3),
              )
            : AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                decoration: BoxDecoration(
                  shape: _recording ? BoxShape.rectangle : BoxShape.circle,
                  borderRadius: _recording ? BorderRadius.circular(8) : null,
                  color: Colors.redAccent,
                ),
              ),
      ),
    );
  }
}
