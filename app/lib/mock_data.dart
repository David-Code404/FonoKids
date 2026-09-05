import 'api_client.dart';

/// Datos de muestra ESTÁTICOS -- se usan como respaldo cuando no hay
/// servidor corriendo, solo para poder ver el diseño de las pantallas sin
/// depender del backend. No representan datos reales.
class MockData {
  /// Mientras esto sea `true`, las pantallas ni intentan hablar con el
  /// servidor -- muestran los datos de muestra directamente. Poner en
  /// `false` cuando el servidor esté prendido y se quiera ver todo real.
  static const bool previewOnly = true;

  static DatasetStats stats() {
    final palabras = [
      ('camba_de_mierda', 500, 480),
      ('ciego_de_mierda', 500, 465),
      ('asqueroso', 450, 420),
      ('colla_y_mierda', 397, 360),
      ('de_esta_no_te_salvas', 372, 300),
      ('eres_una_rata', 344, 310),
      ('te_voy_a_matar', 320, 290),
      ('enano', 312, 280),
      ('engendro', 298, 250),
      ('estorbo', 275, 240),
      ('feo', 260, 230),
      ('torpe', 245, 200),
      ('estúpido', 238, 190),
      ('inútil', 220, 180),
      ('cállate_la_boca', 215, 170),
      ('me_las_vas_a_pagar', 208, 160),
      ('te_voy_a_encontrar', 196, 150),
      ('cuidate_las_espaldas', 184, 140),
      ('sos_un_estorbo', 176, 130),
      ('desgraciado', 168, 120),
      ('maldito', 160, 115),
      ('rata_inmunda', 152, 105),
      ('basura_humana', 145, 100),
      ('no_sirves_para_nada', 138, 95),
      ('vas_a_sufrir', 130, 88),
      ('te_voy_a_destruir', 122, 80),
      ('sos_un_fracaso', 114, 72),
      ('idiota', 108, 66),
      ('imbécil', 100, 60),
      ('miserable', 92, 52),
      ('cobarde_de_mierda', 84, 44),
      ('animal', 76, 38),
      ('sucio', 68, 30),
      ('repugnante', 60, 24),
      ('vergüenza_ajena', 52, 18),
      ('sos_un_error', 44, 12),
      ('acabaré_contigo', 36, 8),
      ('ya_vas_a_ver', 28, 4),
    ];
    return DatasetStats(
      goalPerWord: 500,
      totalWords: palabras.length,
      totalClips: palabras.fold(0, (sum, p) => sum + p.$2),
      totalProcessed: palabras.fold(0, (sum, p) => sum + p.$3),
      words: palabras
          .map((p) => WordProgress(word: p.$1, clips: p.$2, processed: p.$3))
          .toList(),
    );
  }

  static List<Recording> recordings() {
    const personas = ['Usuario 1', 'Usuario 2'];
    final datos = [
      ('2026-09-05', 'camba_de_mierda', personas[0]),
      ('2026-09-05', 'ciego_de_mierda', personas[0]),
      ('2026-09-05', 'asqueroso', personas[1]),
      ('2026-09-05', 'te_voy_a_matar', personas[0]),
      ('2026-09-04', 'colla_y_mierda', personas[0]),
      ('2026-09-04', 'de_esta_no_te_salvas', personas[1]),
      ('2026-09-04', 'eres_una_rata', personas[0]),
      ('2026-09-03', 'enano', personas[0]),
      ('2026-09-03', 'engendro', personas[1]),
      ('2026-09-03', 'estorbo', personas[0]),
      ('2026-09-03', 'feo', personas[0]),
      ('2026-09-02', 'torpe', personas[1]),
      ('2026-09-02', 'estúpido', personas[0]),
      ('2026-09-02', 'inútil', personas[0]),
      ('2026-09-01', 'cállate_la_boca', personas[1]),
      ('2026-09-01', 'me_las_vas_a_pagar', personas[0]),
      ('2026-08-31', 'te_voy_a_encontrar', personas[0]),
      ('2026-08-31', 'cuidate_las_espaldas', personas[1]),
      ('2026-08-31', 'sos_un_estorbo', personas[0]),
    ];
    return datos
        .map((d) => Recording(date: d.$1, word: d.$2, person: d.$3, count: 1))
        .toList();
  }
}
