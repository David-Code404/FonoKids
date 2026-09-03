import 'package:flutter/material.dart';

import 'api_client.dart';
import 'app_theme.dart';

/// Lista de todas las frases que forman parte del dataset, con buscador.
class PhrasesScreen extends StatefulWidget {
  const PhrasesScreen({super.key});

  @override
  State<PhrasesScreen> createState() => _PhrasesScreenState();
}

class _PhrasesScreenState extends State<PhrasesScreen> {
  late Future<DatasetStats> _statsFuture;
  String _query = '';

  @override
  void initState() {
    super.initState();
    _statsFuture = _load();
  }

  Future<DatasetStats> _load() async {
    final serverUrl = await ApiClient.getServerUrl();
    return ApiClient.getDatasetStats(serverUrl);
  }

  Future<void> _refresh() async {
    setState(() {
      _statsFuture = _load();
    });
    await _statsFuture;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Frases'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded),
            tooltip: 'Actualizar',
            onPressed: _refresh,
          ),
        ],
      ),
      body: FutureBuilder<DatasetStats>(
        future: _statsFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return ErrorState(message: '${snapshot.error}', onRetry: _refresh);
          }

          final allWords = [...snapshot.data!.words]
            ..sort((a, b) => a.word.compareTo(b.word));

          if (allWords.isEmpty) {
            return const EmptyState(
              icon: Icons.record_voice_over_rounded,
              message: 'Todavía no hay frases en el dataset.',
            );
          }

          final filtered = _query.isEmpty
              ? allWords
              : allWords
                  .where((w) => w.word.replaceAll('_', ' ').contains(_query))
                  .toList();

          return Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 12, 20, 12),
                child: _SearchField(
                  onChanged: (v) => setState(() => _query = v.trim().toLowerCase()),
                ),
              ),
              Expanded(
                child: filtered.isEmpty
                    ? const EmptyState(
                        icon: Icons.search_off_rounded,
                        message: 'Ninguna frase coincide con la búsqueda.',
                      )
                    : RefreshIndicator(
                        onRefresh: _refresh,
                        child: ListView.builder(
                          padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
                          itemCount: filtered.length,
                          itemBuilder: (context, i) {
                            return _PhraseTile(
                              text: filtered[i].word.replaceAll('_', ' '),
                              isLast: i == filtered.length - 1,
                            );
                          },
                        ),
                      ),
              ),
            ],
          );
        },
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
        borderRadius: BorderRadius.circular(12),
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

class _PhraseTile extends StatelessWidget {
  final String text;
  final bool isLast;

  const _PhraseTile({required this.text, required this.isLast});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.only(bottom: isLast ? 0 : 10),
      child: AppCard(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: AppColors.accent2.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(9),
              ),
              child: const Icon(Icons.record_voice_over_rounded,
                  color: AppColors.accent2, size: 16),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Text(
                text,
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 14.5,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
