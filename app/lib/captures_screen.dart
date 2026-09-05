import 'package:flutter/material.dart';

import 'api_client.dart';
import 'app_theme.dart';
import 'mock_data.dart';

/// Lista de sesiones de grabación agrupadas por fecha: quién grabó qué frase
/// y cuándo, sin imágenes.
class CapturesScreen extends StatefulWidget {
  const CapturesScreen({super.key});

  @override
  State<CapturesScreen> createState() => _CapturesScreenState();
}

class _CapturesScreenState extends State<CapturesScreen> {
  late Future<List<Recording>> _recordingsFuture;
  String? _selectedPerson;

  @override
  void initState() {
    super.initState();
    _recordingsFuture = _load();
  }

  Future<List<Recording>> _load() async {
    if (MockData.previewOnly) return MockData.recordings();
    try {
      final serverUrl = await ApiClient.getServerUrl();
      return await ApiClient.getRecordings(serverUrl);
    } catch (_) {
      // Sin servidor corriendo -- muestra datos de muestra para ver el diseño.
      return MockData.recordings();
    }
  }

  Future<void> _refresh() async {
    setState(() {
      _recordingsFuture = _load();
    });
    await _recordingsFuture;
  }

  @override
  Widget build(BuildContext context) {
    return AppBackground(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: SafeArea(
          bottom: false,
          child: Column(
            children: [
              const ScreenHeader(
                title: 'Capturas',
                subtitle: 'Historial de sesiones grabadas',
                icon: Icons.event_note_rounded,
              ),
              Expanded(
                child: FutureBuilder<List<Recording>>(
                  future: _recordingsFuture,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState != ConnectionState.done) {
                      return const Center(child: CircularProgressIndicator());
                    }
                    if (snapshot.hasError) {
                      return ErrorState(message: '${snapshot.error}', onRetry: _refresh);
                    }

                    final recordings = snapshot.data!;
                    if (recordings.isEmpty) {
                      return const EmptyState(
                        icon: Icons.event_busy_rounded,
                        message: 'Todavía no hay grabaciones.',
                      );
                    }

                    final people = recordings.map((r) => r.person).toSet().toList()
                      ..sort();

                    final visible = _selectedPerson == null
                        ? recordings
                        : recordings.where((r) => r.person == _selectedPerson).toList();

                    final grouped = <String, List<Recording>>{};
                    for (final r in visible) {
                      grouped.putIfAbsent(r.date, () => []).add(r);
                    }
                    final dates = grouped.keys.toList()..sort((a, b) => b.compareTo(a));

                    return RefreshIndicator(
                      onRefresh: _refresh,
                      child: ListView(
                        padding: const EdgeInsets.fromLTRB(20, 12, 20, 120),
                        children: [
                          _SummaryStrip(
                            totalSesiones: recordings.length,
                            totalDias: recordings.map((r) => r.date).toSet().length,
                            totalPersonas: people.length,
                          ),
                          const SizedBox(height: 18),
                          const SectionLabel('FILTRAR POR PERSONA'),
                          const SizedBox(height: 10),
                          _PersonFilterRow(
                            people: people,
                            selected: _selectedPerson,
                            onSelected: (p) => setState(() => _selectedPerson = p),
                          ),
                          const SizedBox(height: 22),
                          const SectionLabel('LÍNEA DE TIEMPO'),
                          const SizedBox(height: 12),
                          if (dates.isEmpty)
                            const Padding(
                              padding: EdgeInsets.only(top: 20),
                              child: EmptyState(
                                icon: Icons.person_search_rounded,
                                message: 'Esta persona todavía no grabó ninguna frase.',
                              ),
                            )
                          else
                            for (var i = 0; i < dates.length; i++)
                              _TimelineDay(
                                date: dates[i],
                                items: grouped[dates[i]]!,
                                isLast: i == dates.length - 1,
                              ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SummaryStrip extends StatelessWidget {
  final int totalSesiones;
  final int totalDias;
  final int totalPersonas;

  const _SummaryStrip({
    required this.totalSesiones,
    required this.totalDias,
    required this.totalPersonas,
  });

  @override
  Widget build(BuildContext context) {
    return HeroCard(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 18),
      child: Row(
        children: [
          Expanded(child: _stat(Icons.videocam_rounded, '$totalSesiones', 'Sesiones')),
          _divider(),
          Expanded(child: _stat(Icons.calendar_month_rounded, '$totalDias', 'Días activos')),
          _divider(),
          Expanded(child: _stat(Icons.groups_rounded, '$totalPersonas', 'Personas')),
        ],
      ),
    );
  }

  Widget _divider() => Container(
        width: 1,
        height: 36,
        color: Colors.white.withValues(alpha: 0.25),
      );

  Widget _stat(IconData icon, String value, String label) {
    return Column(
      children: [
        Icon(icon, color: Colors.white, size: 18),
        const SizedBox(height: 6),
        Text(
          value,
          style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w800),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          style: TextStyle(color: Colors.white.withValues(alpha: 0.85), fontSize: 10.5),
        ),
      ],
    );
  }
}

class _PersonFilterRow extends StatelessWidget {
  final List<String> people;
  final String? selected;
  final ValueChanged<String?> onSelected;

  const _PersonFilterRow({
    required this.people,
    required this.selected,
    required this.onSelected,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 34,
      child: ListView(
        scrollDirection: Axis.horizontal,
        children: [
          _chip(context, label: 'Todos', value: null),
          for (final p in people)
            Padding(
              padding: const EdgeInsets.only(left: 8),
              child: _chip(context, label: p.replaceAll('-', ' '), value: p),
            ),
        ],
      ),
    );
  }

  Widget _chip(BuildContext context, {required String label, required String? value}) {
    final isSelected = selected == value;
    return GestureDetector(
      onTap: () => onSelected(value),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
        decoration: BoxDecoration(
          gradient: isSelected ? AppColors.heroGradient : null,
          color: isSelected ? null : AppColors.surface,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: isSelected ? Colors.transparent : AppColors.border),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : AppColors.textSecondary,
            fontSize: 12.5,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
}

class _TimelineDay extends StatelessWidget {
  final String date;
  final List<Recording> items;
  final bool isLast;

  const _TimelineDay({required this.date, required this.items, required this.isLast});

  @override
  Widget build(BuildContext context) {
    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Column(
            children: [
              Container(
                width: 14,
                height: 14,
                margin: const EdgeInsets.only(top: 4),
                decoration: BoxDecoration(
                  gradient: AppColors.heroGradient,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: AppColors.accent.withValues(alpha: 0.5),
                      blurRadius: 8,
                    ),
                  ],
                ),
              ),
              if (!isLast)
                Expanded(
                  child: Container(
                    width: 2,
                    margin: const EdgeInsets.symmetric(vertical: 4),
                    color: AppColors.border,
                  ),
                ),
            ],
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(bottom: 22),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _DateHeader(date: date, count: items.length),
                  const SizedBox(height: 10),
                  AppCard(
                    padding: EdgeInsets.zero,
                    child: Column(
                      children: [
                        for (var j = 0; j < items.length; j++)
                          _RecordingTile(
                            recording: items[j],
                            isLast: j == items.length - 1,
                          ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _DateHeader extends StatelessWidget {
  final String date;
  final int count;

  const _DateHeader({required this.date, required this.count});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(
          date,
          style: const TextStyle(
            color: AppColors.textPrimary,
            fontSize: 14,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(width: 8),
        Text(
          '· $count ${count == 1 ? 'frase' : 'frases'}',
          style: const TextStyle(color: AppColors.textMuted, fontSize: 12.5),
        ),
      ],
    );
  }
}

const _avatarPalette = [
  AppColors.accent,
  AppColors.accent2,
  AppColors.accent3,
  Color(0xFFFF6B81),
  Color(0xFF5AA9FF),
];

class _RecordingTile extends StatelessWidget {
  final Recording recording;
  final bool isLast;

  const _RecordingTile({required this.recording, required this.isLast});

  String get _initials {
    final name = recording.person.replaceAll('-', ' ').trim();
    if (name.isEmpty || name == 'desconocido') return '?';
    final parts = name.split(' ').where((p) => p.isNotEmpty).toList();
    if (parts.length == 1) return parts.first.substring(0, 1).toUpperCase();
    return (parts.first.substring(0, 1) + parts.last.substring(0, 1)).toUpperCase();
  }

  Color get _avatarColor {
    final idx = recording.person.hashCode.abs() % _avatarPalette.length;
    return _avatarPalette[idx];
  }

  @override
  Widget build(BuildContext context) {
    final color = _avatarColor;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
      decoration: isLast
          ? null
          : const BoxDecoration(
              border: Border(bottom: BorderSide(color: AppColors.border, width: 1)),
            ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 17,
            backgroundColor: color.withValues(alpha: 0.18),
            child: Text(
              _initials,
              style: TextStyle(
                color: color,
                fontWeight: FontWeight.w700,
                fontSize: 12.5,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  recording.word.replaceAll('_', ' '),
                  style: const TextStyle(
                    color: AppColors.textPrimary,
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 2),
                Text(
                  recording.person.replaceAll('-', ' '),
                  style: const TextStyle(color: AppColors.textSecondary, fontSize: 12),
                ),
              ],
            ),
          ),
          IconButton(
            icon: Icon(Icons.photo_camera_rounded, color: color, size: 20),
            tooltip: 'Ver captura',
            onPressed: () => _showCaptureDialog(context, color),
          ),
        ],
      ),
    );
  }

  void _showCaptureDialog(BuildContext context, Color color) {
    showDialog(
      context: context,
      builder: (_) => _CaptureDialog(
        recording: recording,
        initials: _initials,
        color: color,
      ),
    );
  }
}

/// Diálogo que muestra la "captura" de una sesión: quién dijo la frase, con
/// una imagen de referencia de la persona (placeholder mientras no haya
/// fotos reales conectadas al backend) y el detalle de la toma.
class _CaptureDialog extends StatelessWidget {
  final Recording recording;
  final String initials;
  final Color color;

  const _CaptureDialog({
    required this.recording,
    required this.initials,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final personName = recording.person.replaceAll('-', ' ').trim();
    final displayName = personName.isEmpty || personName == 'desconocido'
        ? 'Persona no identificada'
        : personName;

    return Dialog(
      backgroundColor: Colors.transparent,
      insetPadding: const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(22),
        child: Container(
          width: 340,
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
                    aspectRatio: 4 / 3,
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
                          Icon(
                            Icons.person_rounded,
                            size: 96,
                            color: Colors.white.withValues(alpha: 0.35),
                          ),
                          Container(
                            width: 74,
                            height: 74,
                            alignment: Alignment.center,
                            decoration: BoxDecoration(
                              color: Colors.white.withValues(alpha: 0.16),
                              shape: BoxShape.circle,
                              border: Border.all(color: Colors.white.withValues(alpha: 0.4), width: 2),
                            ),
                            child: Text(
                              initials,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 26,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ),
                          Positioned(
                            bottom: 10,
                            left: 10,
                            child: Container(
                              padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                              decoration: BoxDecoration(
                                color: Colors.black.withValues(alpha: 0.45),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: const Text(
                                'Imagen de muestra -- sin foto real todavía',
                                style: TextStyle(color: Colors.white70, fontSize: 9.5),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  Positioned(
                    top: 8,
                    right: 8,
                    child: IconButton(
                      icon: const Icon(Icons.close_rounded, color: Colors.white),
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ),
                ],
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      recording.word.replaceAll('_', ' '),
                      style: const TextStyle(
                        color: AppColors.textPrimary,
                        fontSize: 19,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 10),
                    _detailRow(Icons.person_outline_rounded, displayName),
                    const SizedBox(height: 6),
                    _detailRow(Icons.calendar_today_rounded, recording.date),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _detailRow(IconData icon, String text) {
    return Row(
      children: [
        Icon(icon, size: 15, color: AppColors.textMuted),
        const SizedBox(width: 8),
        Text(text, style: const TextStyle(color: AppColors.textSecondary, fontSize: 13)),
      ],
    );
  }
}
