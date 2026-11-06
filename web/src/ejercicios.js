// Ejercicios de calentamiento (versión de PRUEBA). Cada ficha dice qué se le
// muestra al chico, cómo se evalúa y cuándo se da por pasado.
//
// Tipos:
//   praxia -> cámara en el navegador (MediaPipe, blendshapes). Se mide un
//             valor de la cara; tiene que superar `deltas` (uno por señal medida; alcanza con una) por encima del
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
    instruccion: "¡Sonríe lo más grande que puedas y mantenlo!", medida: "sonrisa", deltas: [0.3],
  },
  {
    id: "beso", tipo: "praxia", icono: "😗", titulo: "Beso",
    instruccion: "¡Haz un besito con los labios hacia adelante y mantenlo!", medida: "beso", deltas: [0.3],
  },
  {
    id: "boca", tipo: "praxia", icono: "😮", titulo: "Boca abierta",
    instruccion: "¡Abre la boca bien grande, como un bostezo, y mantenlo!", medida: "boca", deltas: [0.35],
  },
  {
    id: "mejillas", tipo: "praxia", icono: "🐡", titulo: "Mejillas infladas",
    instruccion: "¡Infla las mejillas como un globo y mantenlo!", medida: "mejillas", deltas: [0.08, 0.04],
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

// Estimación del esfuerzo (versión de PRUEBA, sin validar con niños ni con un
// especialista). No mide el esfuerzo muscular real: estima qué tan fuerte, qué
// tan sostenido y qué tan parejo fue el gesto respecto al MÁXIMO de esa misma
// persona (el que se mide en el paso 2 de la praxia). Se muestra al chico solo
// como estrellas (1 a 3), nunca como porcentaje. Los cortes son valores
// iniciales a calibrar.
const FUERTE = 0.75; // un instante cuenta como «firme» si llega al 75 % de su máximo

const media = (v) => v.reduce((a, x) => a + x, 0) / Math.max(v.length, 1);
const nivel = (valor, cortes) => (valor >= cortes[1] ? 3 : valor >= cortes[0] ? 2 : 1);

export const ETIQUETAS_ESFUERZO = {
  fuerza: "💪 Fuerza",
  aguante: "⏱️ Aguante",
  parejo: "🔁 Repeticiones parejas",
  voz: "🗣️ Voz fuerte y clara",
};

/// Praxias. `reps`: una entrada por repetición con `fuerza` (intensidad media
/// durante los 3 s, 1 = su máximo) y `firme` (fracción del tiempo en que llegó
/// al 75 % de su máximo o más).
export function esfuerzoPraxia(reps) {
  if (!reps?.length) return null;
  const fuerzas = reps.map((r) => r.fuerza);
  const alto = Math.max(...fuerzas);
  const bajo = Math.min(...fuerzas);
  return {
    fuerza: nivel(media(fuerzas), [0.65, 0.9]),
    aguante: nivel(media(reps.map((r) => r.firme)), [0.5, 0.85]),
    parejo: nivel(alto > 0 ? 1 - (alto - bajo) / alto : 0, [0.7, 0.9]),
  };
}

/// Soplo. No hay máximo personal (la calibración solo mide el ruido de fondo),
/// así que la fuerza se compara contra el umbral de soplo: `rmsMedio` y
/// `umbral` salen del volumen del micrófono; `firme` es la fracción del soplo
/// con volumen de al menos 1,5 veces el umbral.
export function esfuerzoSoplo(rmsMedio, umbral, firme) {
  return {
    fuerza: nivel(rmsMedio / umbral, [1.5, 2.5]),
    aguante: nivel(firme, [0.5, 0.85]),
  };
}

/// Práctica de palabras y frases. Mide qué tan fuerte y clara se oyó la voz
/// frente al ruido de fondo de ESA grabación: el volumen de los tramos más
/// fuertes dividido por el de los más tranquilos (percentil 90 sobre percentil
/// 10 del volumen de cada bloque de audio). No usa un volumen absoluto porque
/// depende de la ganancia del micrófono. `chunks`: bloques Float32 del
/// grabador. Devuelve null si el audio es demasiado corto para medir.
export function esfuerzoVoz(chunks) {
  if (!chunks || chunks.length < 6) return null;
  const rms = chunks.map((c) => {
    let s = 0;
    for (let i = 0; i < c.length; i++) s += c[i] * c[i];
    return Math.sqrt(s / Math.max(c.length, 1));
  }).sort((a, b) => a - b);
  const pct = (p) => rms[Math.floor(p * (rms.length - 1))];
  const voz = pct(0.9);
  const ruido = pct(0.1);
  if (voz < 0.005) return { voz: 1 }; // casi no se oyó nada
  return { voz: nivel(voz / Math.max(ruido, 1e-4), [3, 8]) };
}

export const SOPLO_FIRME = 1.5;
export const GESTO_FIRME = FUERTE;
