// Palabras para la evaluación y para la práctica por niveles (versión de PRUEBA:
// el contenido tiene que revisarlo un fonoaudiólogo). Cada palabra lleva su
// dibujo para el niño que todavía no lee.
//
// Evaluación: el sonido en posición INICIAL, MEDIA y FINAL de la palabra
// (sonido propio) y los SINFONES (dos consonantes seguidas, "precursores").

export const EVALUACIONES = [
  {
    key: "r", titulo: "Sonido R", icono: "🐸",
    palabras: [
      { palabra: "rana", emoji: "🐸", pos: "inicial" },
      { palabra: "naranja", emoji: "🍊", pos: "media" },
      { palabra: "flor", emoji: "🌸", pos: "final" },
    ],
  },
  {
    key: "s", titulo: "Sonido S", icono: "☀️",
    palabras: [
      { palabra: "sol", emoji: "☀️", pos: "inicial" },
      { palabra: "casa", emoji: "🏠", pos: "media" },
      { palabra: "autobús", emoji: "🚌", pos: "final" },
    ],
  },
  {
    key: "l", titulo: "Sonido L", icono: "🌙",
    palabras: [
      { palabra: "luna", emoji: "🌙", pos: "inicial" },
      { palabra: "pelota", emoji: "⚽", pos: "media" },
      { palabra: "caracol", emoji: "🐌", pos: "final" },
    ],
  },
  {
    key: "k", titulo: "Sonido C / K", icono: "🏠",
    palabras: [
      { palabra: "casa", emoji: "🏠", pos: "inicial" },
      { palabra: "boca", emoji: "👄", pos: "media" },
      { palabra: "foca", emoji: "🦭", pos: "media" },
    ],
  },
  {
    key: "g", titulo: "Sonido G suave", icono: "🐱",
    palabras: [
      { palabra: "gato", emoji: "🐱", pos: "inicial" },
      { palabra: "mago", emoji: "🧙", pos: "media" },
      { palabra: "tigre", emoji: "🐯", pos: "media" },
    ],
  },
  {
    key: "sinfonesL", titulo: "Sinfones con L (PL, BL, GL, FL, CL)", icono: "🎈",
    palabras: [
      { palabra: "plato", emoji: "🍽️", pos: "pl" },
      { palabra: "globo", emoji: "🎈", pos: "gl" },
      { palabra: "flor", emoji: "🌸", pos: "fl" },
      { palabra: "clavo", emoji: "📌", pos: "cl" },
    ],
  },
  {
    key: "sinfonesR", titulo: "Sinfones con R (PR, DR, TR, FR, CR)", icono: "🐉",
    palabras: [
      { palabra: "princesa", emoji: "👸", pos: "pr" },
      { palabra: "dragón", emoji: "🐉", pos: "dr" },
      { palabra: "tren", emoji: "🚂", pos: "tr" },
      { palabra: "fresa", emoji: "🍓", pos: "fr" },
      { palabra: "cruz", emoji: "✝️", pos: "cr" },
    ],
  },
];

export const ETIQUETA_POS = {
  inicial: "al principio", media: "en medio", final: "al final",
  pl: "PL", gl: "GL", fl: "FL", cl: "CL", pr: "PR", dr: "DR", tr: "TR", fr: "FR", cr: "CR",
};

// Práctica por niveles: de 1 a 4 sílabas; se pasa al nivel siguiente cuando se
// logra una palabra del nivel. Palabras sueltas, no oraciones.
export const NIVELES_SILABAS = [
  {
    key: "s", titulo: "Sonido S", icono: "☀️",
    niveles: [
      { silabas: 1, palabras: [{ palabra: "sal", emoji: "🧂" }, { palabra: "sol", emoji: "☀️" }] },
      { silabas: 2, palabras: [{ palabra: "sapo", emoji: "🐸" }, { palabra: "sopa", emoji: "🍲" }, { palabra: "silla", emoji: "🪑" }] },
      { silabas: 3, palabras: [{ palabra: "sandía", emoji: "🍉" }, { palabra: "serpiente", emoji: "🐍" }] },
      { silabas: 4, palabras: [{ palabra: "mariposa", emoji: "🦋" }, { palabra: "semáforo", emoji: "🚦" }] },
    ],
  },
  {
    key: "r", titulo: "Sonido R", icono: "🌹",
    niveles: [
      { silabas: 1, palabras: [{ palabra: "rey", emoji: "👑" }, { palabra: "red", emoji: "🥅" }] },
      { silabas: 2, palabras: [{ palabra: "rana", emoji: "🐸" }, { palabra: "rosa", emoji: "🌹" }] },
      { silabas: 3, palabras: [{ palabra: "camarón", emoji: "🦐" }, { palabra: "pirata", emoji: "🏴‍☠️" }] },
      { silabas: 4, palabras: [{ palabra: "mariposa", emoji: "🦋" }, { palabra: "relámpago", emoji: "⚡" }] },
    ],
  },
  {
    key: "l", titulo: "Sonido L", icono: "🌙",
    niveles: [
      { silabas: 1, palabras: [{ palabra: "luz", emoji: "💡" }, { palabra: "sal", emoji: "🧂" }] },
      { silabas: 2, palabras: [{ palabra: "luna", emoji: "🌙" }, { palabra: "lápiz", emoji: "✏️" }] },
      { silabas: 3, palabras: [{ palabra: "pelota", emoji: "⚽" }, { palabra: "paloma", emoji: "🕊️" }] },
      { silabas: 4, palabras: [{ palabra: "elefante", emoji: "🐘" }, { palabra: "teléfono", emoji: "📞" }] },
    ],
  },
];

// Cuánto tiempo hay que esperar para pensar en un profesional (según las notas
// del fonoaudiólogo): lo normal son 10 sesiones (5 si es leve); si en 6 meses no
// mejora puede haber un trastorno.
export const SESIONES_LEVE = 5;
export const SESIONES_NORMAL = 10;
// Cada cuántos ejercicios hechos toca una evaluación intermedia o final.
export const EJERCICIOS_POR_EVALUACION = 5;

// ---------------------------------------------------------------------------
// Práctica por tipo de sonido (propio, sinfones/precursores, directas e inversas).
// Son palabras sueltas, cortas, con su dibujo. `libre`: todas las palabras
// están abiertas desde el principio.
const P = (palabra, emoji) => ({ palabra, emoji });

export const PRACTICA_SECCIONES = [
  {
    titulo: "🎯 Sonido propio: al principio, en medio y al final",
    grupos: [
      {
        key: "pr_r", titulo: "R", icono: "🐸", libre: true,
        niveles: [
          { titulo: "R al principio", palabras: [P("rana", "🐸"), P("rosa", "🌹"), P("rueda", "🛞"), P("ratón", "🐭"), P("regalo", "🎁")] },
          { titulo: "R en medio", palabras: [P("pera", "🍐"), P("cara", "🙂"), P("loro", "🦜"), P("torre", "🗼"), P("tortuga", "🐢"), P("naranja", "🍊"), P("zanahoria", "🥕"), P("capibara", "🦫")] },
          { titulo: "R al final", palabras: [P("mar", "🌊"), P("flor", "🌸"), P("color", "🎨"), P("doctor", "🧑‍⚕️"), P("tractor", "🚜"), P("cantar", "🎤")] },
        ],
      },
      {
        key: "pr_s", titulo: "S", icono: "☀️", libre: true,
        niveles: [
          { titulo: "S al principio", palabras: [P("sol", "☀️"), P("sal", "🧂"), P("sapo", "🐸"), P("sopa", "🍲"), P("silla", "🪑"), P("serpiente", "🐍")] },
          { titulo: "S en medio", palabras: [P("casa", "🏠"), P("oso", "🐻"), P("vaso", "🥛"), P("pasta", "🍝"), P("queso", "🧀"), P("pasto", "🌱")] },
          { titulo: "S al final", palabras: [P("autobús", "🚌"), P("gafas", "👓"), P("paraguas", "☂️"), P("tijeras", "✂️"), P("lunes", "📅")] },
        ],
      },
      {
        key: "pr_l", titulo: "L", icono: "🌙", libre: true,
        niveles: [
          { titulo: "L al principio", palabras: [P("luna", "🌙"), P("luz", "💡"), P("lápiz", "✏️"), P("lupa", "🔍"), P("libro", "📚"), P("lengua", "👅")] },
          { titulo: "L en medio", palabras: [P("pelota", "⚽"), P("helado", "🍦"), P("vela", "🕯️"), P("mula", "🫏"), P("pala", "⛏️"), P("bola", "🏀")] },
          { titulo: "L al final", palabras: [P("caracol", "🐌"), P("papel", "📄"), P("sol", "☀️"), P("farol", "🏮"), P("hotel", "🏨")] },
        ],
      },
      {
        key: "pr_g", titulo: "G suave", icono: "🐱", libre: true,
        niveles: [
          { titulo: "G al principio", palabras: [P("gato", "🐱"), P("gota", "💧"), P("gorra", "🧢"), P("gusano", "🐛"), P("gallina", "🐔"), P("guitarra", "🎸")] },
          { titulo: "G en medio", palabras: [P("mago", "🧙"), P("tigre", "🐯"), P("lagarto", "🦎"), P("amigo", "🧑‍🤝‍🧑"), P("hormiga", "🐜")] },
        ],
      },
      {
        key: "pr_k", titulo: "C / K", icono: "🥄", libre: true,
        niveles: [
          { titulo: "K al principio", palabras: [P("casa", "🏠"), P("copa", "🏆"), P("cuchara", "🥄"), P("coco", "🥥"), P("camisa", "👕"), P("conejo", "🐰")] },
          { titulo: "K en medio", palabras: [P("boca", "👄"), P("vaca", "🐄"), P("foca", "🦭"), P("taco", "🌮"), P("mono", "🐒"), P("zapato", "👟")] },
        ],
      },
    ],
  },
  {
    titulo: "🔗 Sinfones (precursores de la R y la L)",
    grupos: [
      { key: "sf_l", titulo: "Con L: PL, BL, GL, FL, CL", icono: "🎈", libre: true, niveles: [
        { titulo: "PL", palabras: [P("plato", "🍽️"), P("plátano", "🍌"), P("pluma", "🪶"), P("planeta", "🪐"), P("playa", "🏖️")] },
        { titulo: "BL", palabras: [P("blusa", "👚"), P("blanco", "⬜"), P("bloque", "🧱"), P("blando", "🧸"), P("tabla", "🪵")] },
        { titulo: "GL", palabras: [P("globo", "🎈"), P("iglesia", "⛪"), P("regla", "📏")] },
        { titulo: "FL", palabras: [P("flor", "🌸"), P("flecha", "🏹"), P("flan", "🍮"), P("flauta", "🎶")] },
        { titulo: "CL", palabras: [P("clavo", "📌"), P("clase", "🏫"), P("clavel", "🌺"), P("clima", "⛅")] },
      ] },
      { key: "sf_r", titulo: "Con R: PR, BR, DR, TR, FR, CR, GR", icono: "🐉", libre: true, niveles: [
        { titulo: "PR", palabras: [P("princesa", "👸"), P("premio", "🏆"), P("primavera", "🌷"), P("primo", "🧒"), P("prado", "🌾")] },
        { titulo: "BR", palabras: [P("brazo", "💪"), P("bruja", "🧙"), P("libro", "📖"), P("cabra", "🐐")] },
        { titulo: "DR", palabras: [P("dragón", "🐉"), P("ladrillo", "🧱"), P("dromedario", "🐪"), P("cuadro", "🖼️")] },
        { titulo: "TR", palabras: [P("tren", "🚂"), P("trigo", "🌾"), P("trompeta", "🎺"), P("tres", "3️⃣")] },
        { titulo: "FR", palabras: [P("fresa", "🍓"), P("fruta", "🍎"), P("frío", "🥶"), P("fresco", "🍃")] },
        { titulo: "CR", palabras: [P("cruz", "✝️"), P("crema", "🧴"), P("cráneo", "💀"), P("cráter", "🌋")] },
        { titulo: "GR", palabras: [P("grillo", "🦗"), P("granja", "🏡"), P("gris", "🩶"), P("grande", "🐘")] },
      ] },
    ],
  },
  {
    titulo: "↔️ Sílabas directas e inversas",
    grupos: [
      { key: "di_s", titulo: "S", icono: "☀️", libre: true, niveles: [
        { titulo: "Directas (consonante + vocal): sa, so, se", palabras: [P("sapo", "🐸"), P("sopa", "🍲"), P("sed", "🥤")] },
        { titulo: "Inversas (termina en consonante): es, is, os", palabras: [P("mosca", "🪰"), P("isla", "🏝️"), P("escoba", "🧹")] },
      ] },
      { key: "di_r", titulo: "R", icono: "🐸", libre: true, niveles: [
        { titulo: "Directas: ra, re, ro", palabras: [P("rana", "🐸"), P("pera", "🍐"), P("cara", "🙂")] },
        { titulo: "Inversas: ar, er, or", palabras: [P("puerta", "🚪"), P("torre", "🗼"), P("árbol", "🌳"), P("cuerda", "🪢")] },
      ] },
      { key: "di_l", titulo: "L", icono: "🌙", libre: true, niveles: [
        { titulo: "Directas: la, le, lo", palabras: [P("luna", "🌙"), P("lobo", "🐺"), P("lupa", "🔍")] },
        { titulo: "Inversas: al, el, ol", palabras: [P("falda", "👗"), P("balde", "🪣"), P("palma", "🌴")] },
      ] },
    ],
  },
];

// Qué evaluaciones y práctica van dentro de cada categoría de Aprender, en orden
// lógico: primero el precursor, después el sonido propio. En la R, la L es el
// precursor (antes de la R viene la L); en los sinfones, los de L van antes que los de R.
export const PANELES = {
  rr: {
    titulo: "R",
    bloques: [
      { ev: "r", grupo: "pr_r", directas: "di_r", silabas: "r", nombre: "La R" },
    ],
  },
  s: { titulo: "S", bloques: [{ ev: "s", grupo: "pr_s", directas: "di_s", silabas: "s", nombre: "La S" }] },
  "c-k": { titulo: "C / K", bloques: [{ ev: "k", grupo: "pr_k", nombre: "La C / K" }] },
  g: { titulo: "G suave", bloques: [{ ev: "g", grupo: "pr_g", nombre: "La G suave" }] },
  sinfones: {
    titulo: "Sinfones",
    bloques: [
      { ev: "sinfonesL", grupo: "sf_l", sinfon: true, nombre: "Sinfones con L (precursor)" },
      { ev: "sinfonesR", grupo: "sf_r", sinfon: true, nombre: "Sinfones con R" },
    ],
  },
};

// ---------------------------------------------------------------------------
// Más sonidos (sin modelo entrenado todavía): palabras al principio, en medio y al final.
const grupoExtra = (key, letra, icono, inicial, medio, final) => ({
  key: `pr_${key}`, titulo: letra, icono, libre: true,
  niveles: [
    inicial.length && { titulo: `${letra} al principio`, palabras: inicial.map(([a, b]) => P(a, b)) },
    medio.length && { titulo: `${letra} en medio`, palabras: medio.map(([a, b]) => P(a, b)) },
    final.length && { titulo: `${letra} al final`, palabras: final.map(([a, b]) => P(a, b)) },
  ].filter(Boolean),
});

export const SONIDOS_EXTRA = [
  grupoExtra("p", "P", "🦆", [["pan", "🍞"], ["pato", "🦆"], ["papa", "🥔"], ["piña", "🍍"], ["payaso", "🤡"]], [["sapo", "🐸"], ["copa", "🏆"], ["mapa", "🗺️"], ["tapa", "🥫"], ["zapato", "👟"]], []),
  grupoExtra("b", "B", "👜", [["bus", "🚌"], ["bolsa", "👜"], ["boca", "👄"], ["bebé", "👶"], ["bicicleta", "🚲"]], [["cubo", "🧊"], ["lobo", "🐺"], ["tubo", "🧪"], ["abeja", "🐝"], ["jabón", "🧼"]], []),
  grupoExtra("t", "T", "🍅", [["taza", "☕"], ["toro", "🐂"], ["tigre", "🐯"], ["tomate", "🍅"], ["teléfono", "📞"]], [["pato", "🦆"], ["nota", "🎵"], ["moto", "🏍️"], ["pelota", "⚽"], ["zapato", "👟"]], [["robot", "🤖"]]),
  grupoExtra("d", "D", "🐬", [["delfín", "🐬"], ["dedo", "☝️"], ["dulce", "🍬"], ["dinosaurio", "🦖"]], [["nido", "🪺"], ["helado", "🍦"], ["pedal", "🦶"], ["cuadro", "🖼️"]], [["red", "🥅"], ["pared", "🧱"], ["ciudad", "🏙️"]]),
  grupoExtra("f", "F", "🦭", [["foca", "🦭"], ["foto", "📷"], ["fuego", "🔥"], ["fantasma", "👻"]], [["café", "☕"], ["sofá", "🛋️"], ["jirafa", "🦒"], ["elefante", "🐘"]], []),
  grupoExtra("m", "M", "🐒", [["mano", "✋"], ["mamá", "👩"], ["mono", "🐒"], ["moto", "🏍️"], ["mariposa", "🦋"]], [["cama", "🛏️"], ["humo", "💨"], ["limón", "🍋"], ["camisa", "👕"], ["tomate", "🍅"]], [["álbum", "📒"]]),
  grupoExtra("n", "N", "🚀", [["nave", "🚀"], ["nido", "🪺"], ["nariz", "👃"], ["naranja", "🍊"]], [["luna", "🌙"], ["pino", "🌲"], ["banana", "🍌"], ["manzana", "🍎"]], [["pan", "🍞"], ["balón", "⚽"], ["avión", "✈️"], ["camión", "🚚"]]),
  grupoExtra("j", "J", "🧃", [["jugo", "🧃"], ["jabón", "🧼"], ["jirafa", "🦒"], ["jardín", "🌻"]], [["ojo", "👁️"], ["caja", "📦"], ["oreja", "👂"], ["ventana", "🪟"]], [["reloj", "⌚"]]),
  grupoExtra("z", "Z", "👟", [["zorro", "🦊"], ["zapato", "👟"], ["zanahoria", "🥕"]], [["taza", "☕"], ["pizza", "🍕"], ["plaza", "🏛️"]], [["luz", "💡"], ["pez", "🐟"], ["lápiz", "✏️"], ["arroz", "🍚"]]),
  grupoExtra("ch", "CH", "🍫", [["chancho", "🐷"], ["chaqueta", "🧥"], ["chocolate", "🍫"], ["chivo", "🐐"]], [["coche", "🚙"], ["cuchara", "🥄"], ["noche", "🌃"], ["mochila", "🎒"]], []),
  grupoExtra("ll", "LL", "🔑", [["llave", "🔑"], ["lluvia", "🌧️"], ["llama", "🦙"]], [["pollo", "🍗"], ["silla", "🪑"], ["calle", "🛣️"], ["gallina", "🐔"]], []),
  grupoExtra("y", "Y", "🛥️", [["yo-yo", "🪀"], ["yate", "🛥️"], ["yema", "🥚"], ["yogur", "🥣"]], [["rayo", "⚡"], ["joya", "💎"], ["payaso", "🤡"], ["mayonesa", "🥫"]], []),
  grupoExtra("ni", "Ñ", "🧒", [], [["uña", "💅"], ["año", "🎂"], ["piña", "🍍"], ["baño", "🛁"], ["niño", "🧒"], ["araña", "🕷️"]], []),
];
// La L ya existe como grupo de práctica (pr_l): se suma a la lista de sonidos.
SONIDOS_EXTRA.push(PRACTICA_SECCIONES[0].grupos.find((g) => g.key === "pr_l"));
SONIDOS_EXTRA.forEach((g) => {
  PANELES[`x_${g.key.slice(3)}`] = { titulo: g.titulo, bloques: [{ grupo: g.key, nombre: `La ${g.titulo}` }] };
});
PRACTICA_SECCIONES.push({ titulo: "Más sonidos", grupos: SONIDOS_EXTRA });

// Orden de los sonidos de Aprender, de los más fáciles a los más difíciles (según cuándo se
// adquieren en el habla infantil). Es una propuesta que la fonoaudióloga debe revisar.
export const ORDEN_SONIDOS = [
  "m", "p", "n", "t", "b", "d", "f", "c-k", "g", "ch", "ll", "y", "ni", "l", "j", "z", "s", "rr", "sinfones",
];

// ---------------------------------------------------------------------------
// Evaluaciones de CADA sonido (previa, intermedia al lograr 5 palabras y final al lograr todas).
const GRUPO_DE_EVAL = { r: "pr_r", s: "pr_s", l: "pr_l", k: "pr_k", g: "pr_g", sinfonesL: "sf_l", sinfonesR: "sf_r" };
export const grupoDeEval = (key) => GRUPO_DE_EVAL[key] || `pr_${key}`;

function buscarGrupo(grupoKey) {
  for (const sec of PRACTICA_SECCIONES) for (const g of sec.grupos) if (g.key === grupoKey) return g;
  return null;
}

/// Todas las palabras de un sonido, sin repetir.
export function palabrasDeGrupo(grupoKey) {
  const g = buscarGrupo(grupoKey);
  if (!g) return [];
  const vistas = new Set();
  return g.niveles.flatMap((n) => n.palabras.map((p) => p.palabra)).filter((w) => !vistas.has(w) && vistas.add(w));
}

/// La evaluación de un sonido: la escrita a mano si existe; si no, 3 palabras sacadas de su práctica
/// (una al principio, una en medio y una al final, cuando el sonido las tiene).
export function evaluacionDeGrupo(grupoKey) {
  const key = Object.keys(GRUPO_DE_EVAL).find((k) => GRUPO_DE_EVAL[k] === grupoKey) || grupoKey.replace(/^pr_/, "");
  const hecha = EVALUACIONES.find((e) => e.key === key);
  if (hecha) return hecha;
  const g = buscarGrupo(grupoKey);
  if (!g) return null;
  const pos = (titulo) => (/principio/i.test(titulo) ? "inicial" : /final/i.test(titulo) ? "final" : "media");
  const elegidas = [];
  for (const nv of g.niveles) if (nv.palabras[0] && elegidas.length < 3) elegidas.push({ ...nv.palabras[0], pos: pos(nv.titulo) });
  for (const nv of g.niveles) for (const p of nv.palabras) {
    if (elegidas.length >= 3) break;
    if (!elegidas.some((x) => x.palabra === p.palabra)) elegidas.push({ ...p, pos: pos(nv.titulo) });
  }
  return { key, titulo: `Sonido ${g.titulo}`, icono: g.icono, palabras: elegidas };
}
