// Emoji representativo por palabra objetivo -- para que cada tarjeta se
// reconozca de un vistazo (un chico que todavía no lee bien texto largo sí
// reconoce un dibujo). ICON_FALLBACK si la palabra no está mapeada (ej. una
// palabra nueva que todavía no se agregó acá).
export const WORD_ICONS = {
  perro: "🐶",
  rosa: "🌹",
  ratón: "🐭",
  raton: "🐭",
  carro: "🚗",
  mariposa: "🦋",
  tortuga: "🐢",
  flor: "🌸",
  plátano: "🍌",
  platano: "🍌",
  globo: "🎈",
  tigre: "🐯",
  reloj: "⏰",
  zapato: "👟",
  camión: "🚚",
  camion: "🚚",
  lápiz: "✏️",
  lapiz: "✏️",
  tenedor: "🍴",
  pelota: "⚽",
  tren: "🚂",
  fresa: "🍓",
  rueda: "🛞",
  sombrero: "🎩",
  bicicleta: "🚲",
};

export const ICON_FALLBACK = "🗣";

export function iconFor(word) {
  const key = (word || "").toLowerCase().trim();
  return WORD_ICONS[key] || ICON_FALLBACK;
}
