import 'package:flutter/material.dart';

import 'api_client.dart';
import 'app_theme.dart';
import 'mock_data.dart';

/// Cuántas veces se dijo una frase de riesgo, quién la dijo y cuándo fue la
/// última vez -- a diferencia de Capturas (que muestra sesión por sesión),
/// acá se agrupa por frase.
class _PhraseAgg {
  final String word;
  final int count;
  final String lastDate;
  final List<String> people;

  _PhraseAgg({
    required this.word,
    required this.count,
    required this.lastDate,
    required this.people,
  });
}

enum _SortMode { mostSaid, alphabetical, recent }

const _phraseAvatarPalette = [
  AppColors.accent,
  AppColors.accent2,
  AppColors.accent3,
  Color(0xFFFF6B81),
  Color(0xFF5AA9FF),
];

/// Lista de las frases de riesgo que efectivamente se dijeron (no el
/// progreso de grabación del dataset), con buscador y orden.
class PhrasesScreen extends StatefulWidget {
  const PhrasesScreen({super.key});

  @override
  State<PhrasesScreen> createState() => _PhrasesScreenState();
}

class _PhrasesScreenState extends State<PhrasesScreen> {
  late Future<List<Recording>> _recordingsFuture;
  String _query = '';
  _SortMode _sortMode = _SortMode.mostSaid;

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

  List<_PhraseAgg> _aggregate(List<Recording> recordings) {
    final grouped = <String, List<Recording>>{};
    for (final r in recordings) {
      grouped.putIfAbsent(r.word, () => []).add(r);
    }
    return grouped.entries.map((e) {
      final items = e.value..sort((a, b) => b.date.compareTo(a.date));
      return _PhraseAgg(
        word: e.key,
        count: items.length,
        lastDate: items.first.date,
        people: items.map((r) => r.person).toSet().toList(),
      );
    }).toList();
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
                title: 'Frases',
                subtitle: 'Frases de riesgo detectadas',
                icon: Icons.record_voice_over_rounded,
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 14, 20, 4),
                child: Row(
                  children: [
                    Expanded(
                      child: _SearchField(
                        onChanged: (v) => setState(() => _query = v.trim().toLowerCase()),
                      ),
                    ),
                    const SizedBox(width: 10),
                    _SortButton(
                      mode: _sortMode,
                      onChanged: (m) => setState(() => _sortMode = m),
                    ),
                  ],
                ),
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

                    final all = _aggregate(snapshot.data!);
                    if (all.isEmpty) {
                      return const EmptyState(
                        icon: Icons.record_voice_over_rounded,
                        message: 'Todavía no se detectó ninguna frase de riesgo.',
                      );
                    }

                    switch (_sortMode) {
                      case _SortMode.alphabetical:
                        all.sort((a, b) => a.word.compareTo(b.word));
                        break;
                      case _SortMode.mostSaid:
                        all.sort((a, b) => b.count.compareTo(a.count));
                        break;
                      case _SortMode.recent:
                        all.sort((a, b) => b.lastDate.compareTo(a.lastDate));
                        break;
                    }

                    final filtered = _query.isEmpty
                        ? all
                        : all
                            .where((w) => w.word.replaceAll('_', ' ').contains(_query))
                            .toList();

                    if (filtered.isEmpty) {
                      return const EmptyState(
                        icon: Icons.search_off_rounded,
                        message: 'Ninguna frase coincide con la búsqueda.',
                      );
                    }

                    return RefreshIndicator(
                      onRefresh: _refresh,
                      child: GridView.builder(
                        padding: const EdgeInsets.fromLTRB(20, 12, 20, 120),
                        gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
                          maxCrossAxisExtent: 210,
                          mainAxisSpacing: 12,
                          crossAxisSpacing: 12,
                          mainAxisExtent: 168,
                        ),
                        itemCount: filtered.length,
                        itemBuilder: (context, i) {
                          final w = filtered[i];
                          return _PhraseCard(
                            agg: w,
                            color: AppColors.chartPalette[i % AppColors.chartPalette.length],
                          );
                        },
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

class _SearchField extends StatelessWidget {
  final ValueChanged<String> onChanged;

  const _SearchField({required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: TextField(
        onChanged: onChanged,
        style: const TextStyle(color: AppColors.textPrimary, fontSize: 14),
        decoration: const InputDecoration(
          hintText: 'Buscar frase...',
          hintStyle: TextStyle(color: AppColors.textMuted),
          prefixIcon: Icon(Icons.search_rounded, color: AppColors.textMuted, size: 20),
          border: InputBorder.none,
          contentPadding: EdgeInsets.symmetric(vertical: 14),
        ),
      ),
    );
  }
}

class _SortButton extends StatelessWidget {
  final _SortMode mode;
  final ValueChanged<_SortMode> onChanged;

  const _SortButton({required this.mode, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<_SortMode>(
      tooltip: 'Ordenar',
      initialValue: mode,
      onSelected: onChanged,
      color: AppColors.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: const BorderSide(color: AppColors.border),
      ),
      itemBuilder: (context) => [
        _item(_SortMode.mostSaid, Icons.local_fire_department_rounded, 'Más dichas primero'),
        _item(_SortMode.recent, Icons.schedule_rounded, 'Más recientes primero'),
        _item(_SortMode.alphabetical, Icons.sort_by_alpha_rounded, 'Alfabético'),
      ],
      child: Container(
        padding: const EdgeInsets.all(13),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.border),
        ),
        child: const Icon(Icons.swap_vert_rounded, color: AppColors.textSecondary, size: 20),
      ),
    );
  }

  PopupMenuItem<_SortMode> _item(_SortMode value, IconData icon, String label) {
    final selected = value == mode;
    return PopupMenuItem(
      value: value,
      child: Row(
        children: [
          Icon(icon, size: 17, color: selected ? AppColors.accent : AppColors.textMuted),
          const SizedBox(width: 10),
          Text(
            label,
            style: TextStyle(
              color: selected ? AppColors.accent : AppColors.textPrimary,
              fontSize: 13,
              fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}

class _PhraseCard extends StatelessWidget {
  final _PhraseAgg agg;
  final Color color;

  const _PhraseCard({required this.agg, required this.color});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => _showDetail(context),
      child: AppCard(
        padding: const EdgeInsets.fromLTRB(14, 14, 14, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(Icons.record_voice_over_rounded, color: color, size: 17),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: AppColors.danger.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    '${agg.count}x',
                    style: const TextStyle(
                      color: AppColors.danger,
                      fontSize: 11,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(
              agg.word.replaceAll('_', ' '),
              style: const TextStyle(
                color: AppColors.textPrimary,
                fontSize: 14,
                fontWeight: FontWeight.w700,
                height: 1.2,
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
            const Spacer(),
            Row(
              children: [
                Icon(Icons.schedule_rounded, size: 12, color: AppColors.textMuted),
                const SizedBox(width: 5),
                Expanded(
                  child: Text(
                    'Última vez: ${agg.lastDate}',
                    style: const TextStyle(color: AppColors.textMuted, fontSize: 10.5),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            _PeopleRow(people: agg.people),
          ],
        ),
      ),
    );
  }

  void _showDetail(BuildContext context) {
    showDialog(
      context: context,
      builder: (_) => Dialog(
        backgroundColor: Colors.transparent,
        child: Container(
          width: 320,
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: AppColors.border),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(
                      agg.word.replaceAll('_', ' '),
                      style: const TextStyle(
                        color: AppColors.textPrimary,
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close_rounded, color: AppColors.textMuted),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
              Text(
                'Se dijo ${agg.count} ${agg.count == 1 ? 'vez' : 'veces'} -- '
                'última el ${agg.lastDate}.',
                style: const TextStyle(color: AppColors.textSecondary, fontSize: 12.5),
              ),
              const SizedBox(height: 14),
              const Text(
                'PERSONAS',
                style: TextStyle(
                  color: AppColors.textMuted,
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.6,
                ),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  for (final p in agg.people)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: AppColors.surfaceAlt,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: AppColors.border),
                      ),
                      child: Text(
                        p.replaceAll('-', ' '),
                        style: const TextStyle(color: AppColors.textPrimary, fontSize: 12),
                      ),
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _PeopleRow extends StatelessWidget {
  final List<String> people;

  const _PeopleRow({required this.people});

  Color _colorFor(String p) => _phraseAvatarPalette[p.hashCode.abs() % _phraseAvatarPalette.length];

  String _initial(String p) {
    final n = p.trim();
    return n.isEmpty ? '?' : n.substring(0, 1).toUpperCase();
  }

  @override
  Widget build(BuildContext context) {
    const maxShown = 3;
    final shown = people.take(maxShown).toList();
    final extra = people.length - shown.length;

    return Row(
      children: [
        SizedBox(
          height: 20,
          child: Stack(
            children: [
              for (var i = 0; i < shown.length; i++)
                Positioned(
                  left: i * 14.0,
                  child: CircleAvatar(
                    radius: 10,
                    backgroundColor: AppColors.surface,
                    child: CircleAvatar(
                      radius: 9,
                      backgroundColor: _colorFor(shown[i]).withValues(alpha: 0.85),
                      child: Text(
                        _initial(shown[i]),
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 9,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
        if (shown.isNotEmpty) SizedBox(width: shown.length * 14.0 + 4),
        if (extra > 0)
          Text('+$extra', style: const TextStyle(color: AppColors.textMuted, fontSize: 10.5)),
      ],
    );
  }
}
