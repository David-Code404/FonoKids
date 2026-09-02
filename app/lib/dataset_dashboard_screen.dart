import 'package:flutter/material.dart';

import 'api_client.dart';

/// Pantalla de progreso del dataset: cuántos clips grabados y procesados
/// hay por palabra, con barra de progreso hacia la meta (goal_per_word).
class DatasetDashboardScreen extends StatefulWidget {
  final String serverUrl;

  const DatasetDashboardScreen({super.key, required this.serverUrl});

  @override
  State<DatasetDashboardScreen> createState() => _DatasetDashboardScreenState();
}

class _DatasetDashboardScreenState extends State<DatasetDashboardScreen> {
  late Future<DatasetStats> _statsFuture;

  @override
  void initState() {
    super.initState();
    _statsFuture = ApiClient.getDatasetStats(widget.serverUrl);
  }

  Future<void> _refresh() async {
    setState(() {
      _statsFuture = ApiClient.getDatasetStats(widget.serverUrl);
    });
    await _statsFuture;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Progreso del dataset'),
      ),
      body: FutureBuilder<DatasetStats>(
        future: _statsFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  '${snapshot.error}',
                  style: const TextStyle(color: Colors.white70),
                  textAlign: TextAlign.center,
                ),
              ),
            );
          }

          final stats = snapshot.data!;
          return RefreshIndicator(
            onRefresh: _refresh,
            child: ListView(
              padding: const EdgeInsets.all(16),
              children: [
                _buildSummaryRow(stats),
                const SizedBox(height: 20),
                if (stats.words.isEmpty)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 40),
                    child: Center(
                      child: Text(
                        'No hay ninguna palabra grabada todavía.',
                        style: TextStyle(color: Colors.white54),
                      ),
                    ),
                  )
                else
                  ...stats.words.map((w) => _WordProgressCard(
                        progress: w,
                        goal: stats.goalPerWord,
                      )),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildSummaryRow(DatasetStats stats) {
    return Row(
      children: [
        Expanded(child: _summaryCard('${stats.totalWords}', 'Palabras')),
        const SizedBox(width: 10),
        Expanded(child: _summaryCard('${stats.totalClips}', 'Grabados')),
        const SizedBox(width: 10),
        Expanded(child: _summaryCard('${stats.totalProcessed}', 'Procesados')),
      ],
    );
  }

  Widget _summaryCard(String value, String label) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 14),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1D24),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFF2A2E38)),
      ),
      child: Column(
        children: [
          Text(value,
              style: const TextStyle(
                  color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)),
          const SizedBox(height: 2),
          Text(label, style: const TextStyle(color: Colors.white54, fontSize: 11)),
        ],
      ),
    );
  }
}

class _WordProgressCard extends StatelessWidget {
  final WordProgress progress;
  final int goal;

  const _WordProgressCard({required this.progress, required this.goal});

  @override
  Widget build(BuildContext context) {
    final clipsPct = (progress.clips / goal).clamp(0.0, 1.0);
    final processedPct = (progress.processed / goal).clamp(0.0, 1.0);

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1D24),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFF2A2E38)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(progress.word,
                  style: const TextStyle(
                      color: Colors.white, fontWeight: FontWeight.w600, fontSize: 15)),
              Text(
                '${progress.clips} grabados · ${progress.processed} procesados · meta $goal',
                style: const TextStyle(color: Colors.white54, fontSize: 11),
              ),
            ],
          ),
          const SizedBox(height: 8),
          _bar(clipsPct, const Color(0xFF4F7CFF)),
          const SizedBox(height: 4),
          _bar(processedPct, const Color(0xFF35C46A)),
        ],
      ),
    );
  }

  Widget _bar(double pct, Color color) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(6),
      child: LinearProgressIndicator(
        value: pct,
        minHeight: 7,
        backgroundColor: const Color(0xFF0B0D11),
        valueColor: AlwaysStoppedAnimation<Color>(color),
      ),
    );
  }
}
