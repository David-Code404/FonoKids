// Datos de muestra ESTÁTICOS -- se usan como respaldo TEMPORAL mientras no
// hay dataset real todavía (server sin clips, o /predict sin usar), solo
// para poder ver el diseño completo de las pantallas (Historial/Palabras)
// sin depender del backend. No representan intentos reales de nadie.
//
// Las palabras elegidas son las típicas de logopedia infantil -- las que
// más les cuesta a los chicos (rotacismo con doble erre: "perro", "carro",
// "ratón"; consonante vibrante simple: "flor", "reloj"; combinaciones
// consonánticas: "plátano", "tenedor", "camión") -- no palabras al azar.
const personas = ["Martina", "Lucas", "Sofía"];

// [fecha, palabra, correcta, persona]
const datos = [
  ["2026-09-16", "perro", true, personas[0]],
  ["2026-09-16", "perro", false, personas[0]],
  ["2026-09-16", "carro", true, personas[1]],
  ["2026-09-16", "flor", true, personas[1]],
  ["2026-09-16", "tortuga", false, personas[2]],
  ["2026-09-15", "perro", true, personas[0]],
  ["2026-09-15", "mariposa", true, personas[0]],
  ["2026-09-15", "mariposa", false, personas[1]],
  ["2026-09-15", "ratón", true, personas[2]],
  ["2026-09-15", "carro", false, personas[2]],
  ["2026-09-14", "flor", true, personas[0]],
  ["2026-09-14", "tortuga", true, personas[1]],
  ["2026-09-14", "tortuga", true, personas[1]],
  ["2026-09-14", "ratón", false, personas[0]],
  ["2026-09-13", "perro", true, personas[1]],
  ["2026-09-13", "carro", true, personas[0]],
  ["2026-09-13", "mariposa", true, personas[2]],
];

export function mockRecordings() {
  return datos.map(([date, word, correcta, person], i) => ({
    date,
    word,
    correcta,
    person,
    time: `${10 + (i % 8)}:${(i * 7) % 60 < 10 ? "0" : ""}${(i * 7) % 60}`,
    count: 1,
    class_name: `${word}_${correcta ? "correcto" : "incorrecto"}`,
    thumbnail_file: null,
  }));
}
