import 'package:flutter/material.dart';

import 'api_client.dart';
import 'app_theme.dart';

/// Lista de sesiones de grabación agrupadas por fecha: quién grabó qué frase
/// y cuándo, sin imágenes.
class CapturesScreen extends StatefulWidget {
  const CapturesScreen({super.key});

  @override
  State<CapturesScreen> createState() => _CapturesScreenState();
}

class _CapturesScreenState extends State<CapturesScreen> {
  late Future<List<Recording>> _recordingsFuture;

  @override
  void initState() {
    super.initState();
    _recordingsFuture = _load();
  }

  Future<List<Recording>> _load() async {
    final serverUrl = await ApiClient.getServerUrl();
    return ApiClient.getRecordings(serverUrl);
  }

  Future<void> _refresh() async {
    setState(() {
      _recordingsFuture = _load();
    });
    await _recordingsFuture;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Capturas'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded),
            tooltip: 'Actualizar',
            onPressed: _refresh,
          ),
        ],
      ),
      body: FutureBuilder<List<Recording>>(
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

          final grouped = <String, List<Recording>>{};
          for (final r in recordings) {
            grouped.putIfAbsent(r.date, () => []).add(r);
          }
          final dates = grouped.keys.toList()..sort((a, b) => b.compareTo(a));

          return RefreshIndicator(
            onRefresh: _refresh,
            child: ListView.builder(
              padding: const EdgeInsets.fromLTRB(20, 4, 20, 24),
              itemCount: dates.length,
              itemBuilder: (context, i) {
                final date = dates[i];
                final items = grouped[date]!;
                return Padding(
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
                );
              },
            ),
          );
        },
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
        const Icon(Icons.calendar_today_rounded, size: 14, color: AppColors.accent),
        const SizedBox(width: 8),
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

  @override
  Widget build(BuildContext context) {
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
            backgroundColor: AppColors.accent.withValues(alpha: 0.18),
            child: Text(
              _initials,
              style: const TextStyle(
                color: AppColors.accent,
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
        ],
      ),
    );
  }
}
