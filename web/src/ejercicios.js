// Ejercicios de calentamiento (versión de PRUEBA). Cada ficha dice qué se le
// muestra al chico, cómo se evalúa y cuándo se da por pasado.
//
// Tipos:
//   praxia -> cámara en el navegador (MediaPipe, blendshapes). Se mide un
//             valor de la cara; tiene que superar `delta` por encima del
//             reposo del propio chico (calibración de 2 s) y mantenerse
//             HOLD_MS, `repeticiones` veces.
//   soplo  -> micrófono en el navegador. Se mide el volumen sostenido.
//   voz    -> graba y manda el audio a POST /ejercicio_voz del servidor.
//             tipo "ritmo" (cuenta vueltas p-t-k) o "trabalenguas" (GOP).
//
// Los umbrales (`delta`) son valores iniciales para probar, a calibrar con
// varias personas -- y el contenido tiene que revisarlo un especialista.

export const HOLD_MS = 3000; // cuánto se mantiene el gesto para contar 1 repetición
export const REPETICIONES = 3;

export const EJERCICIOS = [
  {
    id: "sonrisa", tipo: "praxia", icono: "😁", titulo: "Sonrisa grande",
    instruccion: "¡Sonríe lo más grande que puedas y mantenlo!", medida: "sonrisa", delta: 0.3,
  },
  {
    id: "beso", tipo: "praxia", icono: "😗", titulo: "Beso",
    instruccion: "¡Haz un besito con los labios hacia adelante y mantenlo!", medida: "beso", delta: 0.3,
  },
  {
    id: "boca", tipo: "praxia", icono: "😮", titulo: "Boca abierta",
    instruccion: "¡Abre la boca bien grande, como un bostezo, y mantenlo!", medida: "boca", delta: 0.35,
  },
  {
    id: "mejillas", tipo: "praxia", icono: "🐡", titulo: "Mejillas infladas",
    instruccion: "¡Infla las mejillas como un globo y mantenlo!", medida: "mejillas", delta: 0.2,
  },
  {
    id: "vela", tipo: "soplo", icono: "🕯️", titulo: "Soplar la vela",
    instruccion: "Toma aire y sopla suave y largo para apagar la vela.",
  },
  {
    id: "pataka", tipo: "voz", modo: "ritmo", icono: "🥁", titulo: "Pa-ta-ka rápido",
    instruccion: "Repite «pa-ta-ka» seguido y rápido, ¡3 veces o más!",
    texto: "pa ta ka pa ta ka pa ta ka", mostrar: "PA-TA-KA  PA-TA-KA  PA-TA-KA",
  },
  {
    id: "tigres", tipo: "voz", modo: "trabalenguas", icono: "🐯", titulo: "Los tres tigres",
    instruccion: "Dilo despacito y claro.",
    texto: "tres tristes tigres tragaban trigo en un trigal",
    mostrar: "Tres tristes tigres tragaban trigo en un trigal",
  },
  {
    id: "clavito", tipo: "voz", modo: "trabalenguas", icono: "🔨", titulo: "Pablito y el clavito",
    instruccion: "Dilo despacito y claro.",
    texto: "pablito clavó un clavito", mostrar: "Pablito clavó un clavito",
  },
  {
    id: "erre", tipo: "voz", modo: "trabalenguas", icono: "🚂", titulo: "Los carros del ferrocarril",
    instruccion: "Dilo despacito y claro.",
    texto: "rápido ruedan los carros del ferrocarril", mostrar: "Rápido ruedan los carros del ferrocarril",
  },
];

export const ETIQUETA_TIPO = { praxia: "Cara", soplo: "Soplo", voz: "Voz" };
