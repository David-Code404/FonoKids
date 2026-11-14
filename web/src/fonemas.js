// Parte una palabra en sus sonidos (fonemas) para el ejercicio "di cada sonido".
// Cada unidad: { letras: "CH", sonido: "ch" }. La H no suena, así que se salta.
const VOCALES = "aeiouáéíóúü";
const quitarTilde = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "");

export function fonemasDe(palabra) {
  const p = palabra.toLowerCase().replace(/[^a-záéíóúüñ]/g, "");
  const sal = [];
  let i = 0;
  while (i < p.length) {
    const c = p[i];
    const dos = p.slice(i, i + 2);
    const sig = p[i + 1] || "";
    if (dos === "ch") { sal.push({ letras: "CH", sonido: "ch" }); i += 2; continue; }
    if (dos === "ll") { sal.push({ letras: "LL", sonido: "ll" }); i += 2; continue; }
    if (dos === "rr") { sal.push({ letras: "RR", sonido: "rr" }); i += 2; continue; }
    if (dos === "qu") { sal.push({ letras: "QU", sonido: "k" }); i += 2; continue; }
    if (dos === "gu" && "eiéí".includes(p[i + 2] || "")) { sal.push({ letras: "GU", sonido: "g" }); i += 2; continue; }
    if (c === "h") { i += 1; continue; }
    if (c === "c" && "eiéí".includes(sig)) { sal.push({ letras: "C", sonido: "s" }); i += 1; continue; }
    if (c === "c") { sal.push({ letras: "C", sonido: "k" }); i += 1; continue; }
    if (c === "x") { sal.push({ letras: "X", sonido: "ks" }); i += 1; continue; }
    sal.push({ letras: quitarTilde(c).toUpperCase().replace("Ü", "U"), sonido: quitarTilde(c) });
    i += 1;
  }
  return sal;
}

export const esVocal = (l) => VOCALES.includes(l.toLowerCase());

// Cómo se le pide al navegador que "diga" cada sonido (lo más parecido al sonido solo).
const DICE = {
  m: "mmm", n: "nnn", f: "fff", s: "sss", z: "sss", l: "lll", r: "rrr", rr: "rrrr", j: "jjj",
  ch: "chhh", ll: "yyy", y: "yyy", ñ: "ñññ", p: "pe", b: "be", t: "te", d: "de", k: "ke", g: "gue", v: "be", ks: "ks", w: "gu",
};

/// Dice en voz alta (voz del navegador, despacito) un sonido o una palabra.
export function decirEnVoz(texto) {
  try {
    const s = window.speechSynthesis;
    if (!s) return;
    s.cancel();
    const u = new SpeechSynthesisUtterance(texto);
    u.lang = "es-ES";
    u.rate = 0.7;
    const voz = s.getVoices().find((v) => v.lang?.toLowerCase().startsWith("es"));
    if (voz) u.voice = voz;
    s.speak(u);
  } catch { /* sin voz, no pasa nada */ }
}

// Sonidos sacados de los videos de referencia (public/sonidos/<sonido>.wav). Los que no tienen
// archivo (vocales, J) usan la voz del navegador hasta que se agregue su audio.
const ARCHIVO = { "ñ": "ni", v: "b" };
const CON_ARCHIVO = new Set(["m", "p", "n", "t", "b", "d", "f", "k", "g", "ch", "ll", "y", "ni", "l", "s", "r", "rr", "a", "e", "i", "o", "u", "j", "z"]);
let sonandoAhora = null;
export function decirSonido(sonido) {
  const clave = ARCHIVO[sonido] || sonido;
  try { window.speechSynthesis?.cancel(); sonandoAhora?.pause(); } catch { /* nada */ }
  if (CON_ARCHIVO.has(clave)) {
    try {
      const a = new Audio(`/sonidos/${clave}.wav`);
      sonandoAhora = a;
      a.play().catch(() => decirEnVoz(DICE[sonido] || sonido));
      return;
    } catch { /* cae a la voz */ }
  }
  decirEnVoz(DICE[sonido] || sonido);
}

// En las cartas se muestra la letra tal como está en la palabra (una N es "N", no "NNN"): así el niño no confunde cuántas letras tiene.
export const comoSeVe = (u) => u.letras;

// Videos de referencia (cara de quien lo dice) por sonido; los que faltan no muestran botón.
const VIDEOS = {
  m: ["m1", "m2"], p: ["p1", "p2"], n: ["n1", "n2"], t: ["t1", "t2"], b: ["b1", "b2"], d: ["d1", "d2"],
  f: ["f1", "f2"], k: ["c1"], g: ["g1", "g2"], ch: ["ch1", "ch2"], ll: ["ll1", "ll2"], y: ["y1", "y2"],
  "ñ": ["ni1", "ni2"], v: ["b1", "b2"], l: ["l1", "l2"], s: ["s1", "s2"], r: ["r"], a: ["a1", "a2"], e: ["e1", "e2"], i: ["i1", "i2"], o: ["o1", "o2"], u: ["u1", "u2"], z: ["z1", "z2"], j: ["j1", "j2"], rr: ["rr", "rrb1", "rrb2"],
};
export const videosDe = (sonido) => (VIDEOS[sonido] || []).map((v) => `/referencia/${v}.mp4`);

// Forma de la carta de cada sonido, para acordarse de qué tipo es:
// círculo = vocal, triángulo = suena largo (mmm, sss...), cuadrado = suena cortito (p, t, k...),
// rectángulo = dos letras que suenan como un solo sonido (ch, ll, rr, qu, gu).
const CORTOS = new Set(["p", "t", "k", "b", "d", "g", "ch"]);
export function formaDe(u) {
  if (u.letras.length > 1) return "rectangulo"; // dos letras que suenan como un solo sonido (CH, LL, RR, QU, GU)
  if (esVocal(u.letras)) return "circulo";
  return CORTOS.has(u.sonido) ? "cuadrado" : "triangulo";
}

/// Qué sonido(s) de la palabra salieron mal, según el puntaje de cada sonido que devuelve el servidor.
/// Devuelve hasta 2 sonidos (los peores) como [{ letras, indice, score }] o null si no se puede ubicar.
/// Las palabras que empiezan con ñ, y, ll o ch no cuentan su primer sonido: el modelo lo puntúa raro aunque esté bien.
export function sonidosAMejorar(palabra, scores) {
  const unidades = fonemasDe(palabra);
  if (!unidades.length || !Array.isArray(scores) || unidades.length !== scores.length) return null;
  const inestable = /^(ñ|y|ll|ch)/i.test(palabra.trim());
  const malos = [];
  scores.forEach((sc, i) => {
    if (sc == null || sc >= -3.0) return;
    if (i === 0 && inestable) return;
    malos.push({ letras: unidades[i].letras, indice: i, score: sc });
  });
  malos.sort((a, b) => a.score - b.score);
  const vistos = new Set();
  return malos.filter((m) => !vistos.has(m.letras) && vistos.add(m.letras)).slice(0, 2);
}

/// Qué sonido(s) de la palabra salieron BIEN (para elogiar con nombre propio: "¡Muy bien la M!").
/// Prefiere consonantes (las vocales casi siempre salen bien y no dicen nada), con puntaje alto, y devuelve hasta 2 letras.
/// Si la palabra no tiene consonantes buenas, usa las vocales. null si no se puede ubicar.
export function sonidosBienDichos(palabra, scores) {
  const unidades = fonemasDe(palabra);
  if (!unidades.length || !Array.isArray(scores) || unidades.length !== scores.length) return null;
  const inestable = /^(ñ|y|ll|ch)/i.test(palabra.trim());
  const buenos = [];
  scores.forEach((sc, i) => {
    if (sc == null || sc < -1.5) return;
    if (i === 0 && inestable) return;
    buenos.push({ letras: unidades[i].letras, indice: i, score: sc, vocal: esVocal(unidades[i].letras) });
  });
  buenos.sort((a, b) => (a.vocal - b.vocal) || (b.score - a.score));
  const vistos = new Set();
  const unicos = buenos.filter((m) => !vistos.has(m.letras) && vistos.add(m.letras));
  const consonantes = unicos.filter((m) => !m.vocal);
  return (consonantes.length ? consonantes : unicos).slice(0, 2);
}
