// Une lo que se practicó con las palabras de siempre (clips guardados en el servidor) y las palabras
// nuevas de Aprender (tabla ejercicio_intentos), y lo agrupa por sonido en el mismo orden de Aprender.
import { SOUND_CATEGORIES } from "./soundCategories.js";
import { ORDEN_SONIDOS, SONIDOS_EXTRA } from "./palabras.js";
import { getServerUrl } from "./api.js";

export const TITULO_SONIDO = {
  "c-k": "C / K", g: "G suave", rr: "R y RR", sinfones: "Sinfones",
  m: "M", p: "P", n: "N", t: "T", b: "B", d: "D", f: "F", ch: "CH", ll: "LL", y: "Y", ni: "Ñ", l: "L", j: "J", z: "Z", s: "S",
};
const ICONO_SONIDO = Object.fromEntries(SONIDOS_EXTRA.map((g) => [g.key.slice(3), g.icono]));
SOUND_CATEGORIES.forEach((c) => { ICONO_SONIDO[c.key] = c.words[0]?.icon || "🗣️"; });
export const iconoDeSonido = (k) => ICONO_SONIDO[k] || "🗣️";

// palabra de siempre -> sonido (por la categoría a la que pertenece)
const SONIDO_DE_PALABRA = {};
SOUND_CATEGORIES.forEach((c) => c.words.forEach((w) => { SONIDO_DE_PALABRA[w.word] = c.key; }));
export function sonidoDePalabraModelo(word) {
  return SONIDO_DE_PALABRA[word.split("_")[0]] || null;
}

// "pr_r_1_naranja" -> { sonido: "rr", palabra: "naranja" }
const ALIAS = { r: "rr", k: "c-k" };
const SONIDO_EVAL = { r: "rr", s: "s", l: "l", k: "c-k", g: "g", sinfonesL: "sinfones", sinfonesR: "sinfones" };
export function parsearIdPalabra(id) {
  // "fon_pr_m_mano" (ejercicio sonido por sonido) cuenta igual que "pr_m_mano"
  const m = /^(?:fon_)?(pr|sf|di)_([a-z]+)(?:_\d+)?_(.+)$/.exec(id || "");
  if (m) {
    const sonido = m[1] === "sf" ? "sinfones" : (ALIAS[m[2]] || m[2]);
    return { sonido, palabra: m[3] };
  }
  // "test_m_mano" (prueba inicial)
  const t = /^test_([a-z-]+)_(.+)$/.exec(id || "");
  if (t) return { sonido: ALIAS[t[1]] || t[1], palabra: t[2] };
  // "r_rana", "sinfonesL_plato" (evaluaciones previa, intermedia y final)
  const e = /^([A-Za-z]+)_(.+)$/.exec(id || "");
  if (e) {
    const sonido = SONIDO_EVAL[e[1]] || ALIAS[e[1]] || e[1];
    if (ORDEN_SONIDOS.includes(sonido)) return { sonido, palabra: e[2] };
  }
  return null;
}

/// recordings de siempre + palabras nuevas (filas de /ejercicios/resumen con tipo "palabra").
/// Cada elemento lleva `sonido`.
export function unirConPalabrasNuevas(recordings, resumen) {
  const lista = (recordings || []).map((r) => ({ ...r, sonido: sonidoDePalabraModelo(r.word) }));
  for (const f of resumen || []) {
    if (!["palabra", "diagnostico", "evaluacion"].includes(f.tipo)) continue;
    const p = parsearIdPalabra(f.ejercicio_id);
    if (!p) continue;
    const fecha = String(f.ultimo || "").slice(0, 10);
    if (f.bien > 0) lista.push({ word: p.palabra, correcta: true, count: f.bien, date: fecha, person: "web", sonido: p.sonido });
    if (f.intentos - f.bien > 0) lista.push({ word: p.palabra, correcta: false, count: f.intentos - f.bien, date: fecha, person: "web", sonido: p.sonido });
  }
  return lista;
}

export async function getResumenEjercicios() {
  try {
    const res = await fetch(`${getServerUrl().replace(/\/+$/, "")}/ejercicios/resumen`);
    if (!res.ok) return [];
    return (await res.json()).ejercicios || [];
  } catch {
    return [];
  }
}

/// Agrupa elementos que tienen `sonido` en el orden de Aprender (los fáciles primero).
export function agruparPorSonido(items) {
  const g = {};
  for (const it of items) (g[it.sonido || "otros"] ??= []).push(it);
  const claves = Object.keys(g).sort((a, b) => {
    const ia = ORDEN_SONIDOS.indexOf(a), ib = ORDEN_SONIDOS.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  return claves.map((k) => ({ clave: k, titulo: TITULO_SONIDO[k] || "Otras", icono: iconoDeSonido(k), items: g[k] }));
}

// Dibujo de cada palabra nueva de Aprender (las de siempre usan wordIcons.js).
import { PRACTICA_SECCIONES, NIVELES_SILABAS } from "./palabras.js";
import { iconFor as iconoBase } from "./wordIcons.js";
const ICONO_PALABRA = {};
const norm = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
PRACTICA_SECCIONES.forEach((s) => s.grupos.forEach((g) => g.niveles.forEach((nv) => nv.palabras.forEach((p) => { ICONO_PALABRA[norm(p.palabra)] ??= p.emoji; }))));
NIVELES_SILABAS.forEach((g) => g.niveles.forEach((nv) => nv.palabras.forEach((p) => { ICONO_PALABRA[norm(p.palabra)] ??= p.emoji; })));
export const iconoDePalabra = (word) => {
  const base = iconoBase(word);
  return base && base !== "🗣" && base !== "🗣️" ? base : (ICONO_PALABRA[norm(word.split("_")[0])] || base);
};
