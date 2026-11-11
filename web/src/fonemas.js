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

// Cómo se escribe el sonido en pantalla (la N se dice "nnn", no "ene").
const SE_VE = { m: "MMM", n: "NNN", f: "FFF", s: "SSS", z: "SSS", l: "LLL", r: "RRR", rr: "RRRR", j: "JJJ", ch: "CHHH", ll: "LL", y: "Y", "ñ": "ÑÑÑ" };
export const comoSeVe = (u) => SE_VE[u.sonido] || u.letras;

// Videos de referencia (cara de quien lo dice) por sonido; los que faltan no muestran botón.
const VIDEOS = {
  m: ["m1", "m2"], p: ["p1", "p2"], n: ["n1", "n2"], t: ["t1", "t2"], b: ["b1", "b2"], d: ["d1", "d2"],
  f: ["f1", "f2"], k: ["c1"], g: ["g1", "g2"], ch: ["ch1", "ch2"], ll: ["ll1", "ll2"], y: ["y1", "y2"],
  "ñ": ["ni1", "ni2"], l: ["l1", "l2"], s: ["s1", "s2"], r: ["r"], a: ["a1", "a2"], e: ["e1", "e2"], i: ["i1", "i2"], o: ["o1", "o2"], u: ["u1", "u2"], z: ["z1", "z2"], j: ["j1", "j2"], rr: ["rr", "rrb1", "rrb2"],
};
export const videosDe = (sonido) => (VIDEOS[sonido] || []).map((v) => `/referencia/${v}.mp4`);

// Forma de la carta de cada sonido, para acordarse de qué tipo es:
// círculo = vocal, triángulo = suena largo (mmm, sss...), cuadrado = suena cortito (p, t, k...).
const CORTOS = new Set(["p", "t", "k", "b", "d", "g", "ch"]);
export function formaDe(u) {
  if (esVocal(u.letras)) return "circulo";
  return CORTOS.has(u.sonido) ? "cuadrado" : "triangulo";
}
