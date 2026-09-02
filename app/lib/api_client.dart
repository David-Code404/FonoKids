import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

/// Resultado de una predicción, tal como lo devuelve server/main.py.
class PredictionResult {
  final bool valid;
  final String? word;
  final double? prob;
  final int? detected;
  final int? total;
  final Map<String, double>? allProbs;
  final String? reason;

  PredictionResult({
    required this.valid,
    this.word,
    this.prob,
    this.detected,
    this.total,
    this.allProbs,
    this.reason,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      valid: json['valid'] as bool? ?? false,
      word: json['word'] as String?,
      prob: (json['prob'] as num?)?.toDouble(),
      detected: json['detected'] as int?,
      total: json['total'] as int?,
      allProbs: (json['all_probs'] as Map<String, dynamic>?)
          ?.map((k, v) => MapEntry(k, (v as num).toDouble())),
      reason: json['reason'] as String?,
    );
  }
}

/// Progreso de una palabra en el dataset, tal como lo devuelve
/// GET /dataset/stats.
class WordProgress {
  final String word;
  final int clips;
  final int processed;

  WordProgress({required this.word, required this.clips, required this.processed});

  factory WordProgress.fromJson(Map<String, dynamic> json) {
    return WordProgress(
      word: json['word'] as String,
      clips: json['clips'] as int,
      processed: json['processed'] as int,
    );
  }
}

/// Resultado completo de GET /dataset/stats.
class DatasetStats {
  final int goalPerWord;
  final int totalWords;
  final int totalClips;
  final int totalProcessed;
  final List<WordProgress> words;

  DatasetStats({
    required this.goalPerWord,
    required this.totalWords,
    required this.totalClips,
    required this.totalProcessed,
    required this.words,
  });

  factory DatasetStats.fromJson(Map<String, dynamic> json) {
    return DatasetStats(
      goalPerWord: json['goal_per_word'] as int,
      totalWords: json['total_words'] as int,
      totalClips: json['total_clips'] as int,
      totalProcessed: json['total_processed'] as int,
      words: (json['words'] as List<dynamic>)
          .map((w) => WordProgress.fromJson(w as Map<String, dynamic>))
          .toList(),
    );
  }
}

/// Excepción específica para errores de red/servidor, para poder mostrar
/// mensajes claros en la UI en vez de un crash o un error genérico.
class ApiException implements Exception {
  final String message;
  ApiException(this.message);

  @override
  String toString() => message;
}

class ApiClient {
  static const _prefsKey = 'server_url';
  static const defaultUrl = 'http://192.168.1.100:8000';

  static Future<String> getServerUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_prefsKey) ?? defaultUrl;
  }

  static Future<void> setServerUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsKey, url.trim());
  }

  static String _normalize(String url) {
    var u = url.trim();
    if (u.endsWith('/')) {
      u = u.substring(0, u.length - 1);
    }
    return u;
  }

  /// Chequea que el servidor esté vivo y con el modelo cargado.
  static Future<bool> checkHealth(String baseUrl) async {
    try {
      final uri = Uri.parse('${_normalize(baseUrl)}/health');
      final response = await http.get(uri).timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Pide el progreso del dataset (cuántos clips por palabra) para el dashboard.
  static Future<DatasetStats> getDatasetStats(String baseUrl) async {
    final uri = Uri.parse('${_normalize(baseUrl)}/dataset/stats');
    try {
      final response = await http.get(uri).timeout(const Duration(seconds: 10));
      if (response.statusCode != 200) {
        throw ApiException('El servidor respondió con error ${response.statusCode}.');
      }
      final Map<String, dynamic> json = jsonDecode(response.body) as Map<String, dynamic>;
      return DatasetStats.fromJson(json);
    } on SocketException {
      throw ApiException(
        'No se pudo conectar al servidor ($baseUrl). '
        'Revisá que esté prendido y que el celular esté en la misma red WiFi.',
      );
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException('Error inesperado pidiendo el progreso del dataset: $e');
    }
  }

  /// Envía el video grabado al servidor y devuelve la predicción.
  static Future<PredictionResult> predict(String baseUrl, File videoFile) async {
    final uri = Uri.parse('${_normalize(baseUrl)}/predict');

    try {
      final request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('file', videoFile.path));

      final streamedResponse =
          await request.send().timeout(const Duration(seconds: 30));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode != 200) {
        throw ApiException(
          'El servidor respondió con error ${response.statusCode}. '
          'Revisá que el modelo esté cargado ahí.',
        );
      }

      final Map<String, dynamic> json = jsonDecode(response.body) as Map<String, dynamic>;
      return PredictionResult.fromJson(json);
    } on SocketException {
      throw ApiException(
        'No se pudo conectar al servidor ($baseUrl). '
        'Revisá que esté prendido y que el celular esté en la misma red WiFi.',
      );
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException('Error inesperado hablando con el servidor: $e');
    }
  }
}
