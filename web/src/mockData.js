// Datos de muestra ESTÁTICOS -- se usan como respaldo cuando no hay servidor
// corriendo, solo para poder ver el diseño de las pantallas sin depender del
// backend. No representan datos reales. Igual que mock_data.dart.
const personas = ["Usuario 1", "Usuario 2"];

const datos = [
  ["2026-09-05", "camba_de_mierda", personas[0]],
  ["2026-09-05", "ciego_de_mierda", personas[0]],
  ["2026-09-05", "asqueroso", personas[1]],
  ["2026-09-05", "te_voy_a_matar", personas[0]],
  ["2026-09-04", "colla_y_mierda", personas[0]],
  ["2026-09-04", "de_esta_no_te_salvas", personas[1]],
  ["2026-09-04", "eres_una_rata", personas[0]],
  ["2026-09-03", "enano", personas[0]],
  ["2026-09-03", "engendro", personas[1]],
  ["2026-09-03", "estorbo", personas[0]],
  ["2026-09-03", "feo", personas[0]],
  ["2026-09-02", "torpe", personas[1]],
  ["2026-09-02", "estúpido", personas[0]],
  ["2026-09-02", "inútil", personas[0]],
  ["2026-09-01", "cállate_la_boca", personas[1]],
  ["2026-09-01", "me_las_vas_a_pagar", personas[0]],
  ["2026-08-31", "te_voy_a_encontrar", personas[0]],
  ["2026-08-31", "cuidate_las_espaldas", personas[1]],
  ["2026-08-31", "sos_un_estorbo", personas[0]],
];

export function mockRecordings() {
  return datos.map(([date, word, person]) => ({ date, word, person, count: 1 }));
}
