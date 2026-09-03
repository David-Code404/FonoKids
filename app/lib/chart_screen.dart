import 'package:flutter/material.dart';

import 'api_client.dart';
import 'app_theme.dart';

/// Gráfico de barras con la distribución de grabaciones por frase.
class ChartScreen extends StatefulWidget {
  const ChartScreen({super.key});

  @override
  State<ChartScreen> createState() => _ChartScreenState();
}

class _ChartScreenState extends State<ChartScreen> {
  late Future<DatasetStats> _statsFuture;

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
        title: const Text('Gráficas'),
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

          final stats = snapshot.data!;
          final withData = stats.words.where((w) => w.clips > 0).toList()
            ..sort((a, b) => b.clips.compareTo(a.clips));

          if (withData.isEmpty) {
            return const EmptyState(
              icon: Icons.bar_chart_rounded,
              message: 'Todavía no hay frases grabadas.',
            );
          }

          final maxValue = withData.first.clips;

          return RefreshIndicator(
            onRefresh: _refresh,
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
              children: [
                Row(
                  children: [
                    Expanded(
                      child: _StatCard(
                        icon: Icons.record_voice_over_rounded,
                        value: '${withData.length}',
                        label: 'Frases activas',
                        color: AppColors.accent,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _StatCard(
                        icon: Icons.trending_up_rounded,
                        value: '${withData.first.clips}',
                        label: 'Frase líder',
                        color: AppColors.accent2,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                Text(
                  'DISTRIBUCIÓN POR FRASE',
                  style: TextStyle(
                    color: AppColors.textMuted,
                    fontSize: 11.5,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.8,
                  ),
                ),
                const SizedBox(height: 10),
                AppCard(
                  padding: const EdgeInsets.fromLTRB(16, 18, 16, 6),
                  child: Column(
                    children: [
                      for (var i = 0; i < withData.length; i++)
                        _BarRow(
                          label: _prettyLabel(withData[i].word),
                          value: withData[i].clips,
                          maxValue: maxValue,
                          color: AppColors.chartPalette[i % AppColors.chartPalette.length],
                          isLast: i == withData.length - 1,
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
  }

  static String _prettyLabel(String word) => word.replaceAll('_', ' ');
}

class _StatCard extends StatelessWidget {
  final IconData icon;
  final String value;
  final String label;
  final Color color;

  const _StatCard({
    required this.icon,
    required this.value,
    required this.label,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: color, size: 18),
          ),
          const SizedBox(height: 12),
          Text(
            value,
            style: const TextStyle(
              color: AppColors.textPrimary,
              fontSize: 22,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 2),
          Text(label, style: const TextStyle(color: AppColors.textSecondary, fontSize: 12)),
        ],
      ),
    );
  }
}

class _BarRow extends StatelessWidget {
  final String label;
  final int value;
  final int maxValue;
  final Color color;
  final bool isLast;

  const _BarRow({
    required this.label,
    required this.value,
    required this.maxValue,
    required this.color,
    required this.isLast,
  });

  @override
  Widget build(BuildContext context) {
    final fraction = maxValue == 0 ? 0.0 : value / maxValue;
    return Container(
      padding: const EdgeInsets.only(bottom: 14),
      margin: EdgeInsets.only(bottom: isLast ? 0 : 14),
      decoration: isLast
          ? null
          : const BoxDecoration(
              border: Border(bottom: BorderSide(color: AppColors.border, width: 1)),
            ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Text(
                  label,
                  style: const TextStyle(
                    color: AppColors.textPrimary,
                    fontSize: 13.5,
                    fontWeight: FontWeight.w600,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 10),
              Text(
                '$value',
                style: const TextStyle(
                  color: AppColors.textSecondary,
                  fontSize: 12.5,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          LayoutBuilder(
            builder: (context, constraints) {
              return Stack(
                children: [
                  Container(
                    height: 8,
                    decoration: BoxDecoration(
                      color: AppColors.surfaceAlt,
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 500),
                    curve: Curves.easeOutCubic,
                    height: 8,
                    width: constraints.maxWidth * fraction,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(colors: [color.withValues(alpha: 0.6), color]),
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                ],
              );
            },
          ),
        ],
      ),
    );
  }
}
