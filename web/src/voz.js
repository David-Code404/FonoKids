// Voz de la mascota: dice en voz alta las instrucciones y los ánimos (para los niños que todavía no leen).
// Usa la voz del navegador; se puede silenciar con el botón 🔊 y se acuerda de la elección.
import { getServerUrl } from "./api.js";

const CLAVE = "fonokids_voz";
let audioActual = null;
const oyentes = new Set();

export function vozActiva() {
  try { return localStorage.getItem(CLAVE) !== "0"; } catch { return true; }
}

export function ponerVoz(activa) {
  try { localStorage.setItem(CLAVE, activa ? "1" : "0"); } catch { /* nada */ }
  if (!activa) callar();
  oyentes.forEach((f) => f(activa));
}

export function alCambiarVoz(f) {
  oyentes.add(f);
  return () => oyentes.delete(f);
}

export function callar() {
  try { window.speechSynthesis?.cancel(); } catch { /* nada */ }
  try { if (audioActual) { audioActual.pause(); audioActual = null; } } catch { /* nada */ }
}

// Saca los dibujitos (emojis) para que la voz no los deletree.
const sinEmojis = (t) => t.replace(/[\p{Extended_Pictographic}️‍]/gu, "").replace(/\s+/g, " ").trim();

// Voz del navegador (robótica): YA NO se usa para hablar; se deja por si algún día se necesita.
// eslint-disable-next-line no-unused-vars
function hablarConElNavegador(limpio) {
  try {
    const s = window.speechSynthesis;
    if (!s) return;
    s.cancel();
    const u = new SpeechSynthesisUtterance(limpio);
    u.lang = "es-ES";
    u.rate = 0.88;
    u.pitch = 1.2; // un poco más aguda: voz amable para chicos
    const voz = s.getVoices().find((v) => v.lang?.toLowerCase().startsWith("es"));
    if (voz) u.voice = voz;
    s.speak(u);
  } catch { /* sin voz, no pasa nada */ }
}

// Voz natural de Fofo: la genera el servidor (Piper, sin internet). Si no responde, cae a la del navegador.
export function hablar(texto, { forzar = false, fuerte = false, alTerminar = null } = {}) {
  if (!forzar && !vozActiva()) return;
  const limpio = sinEmojis(texto || "");
  if (!limpio) return;
  callar();
  try {
    const a = new Audio(`${getServerUrl()}/voz?texto=${encodeURIComponent(limpio)}${fuerte ? "&fuerte=1" : ""}`);
    a.volume = 1;
    audioActual = a;
    // Si el servidor no responde, Fofo se queda callada: nunca usa la voz robótica del navegador.
    a.onerror = () => { if (audioActual === a) audioActual = null; };
    if (alTerminar) a.onended = () => { if (audioActual === a) { audioActual = null; alTerminar(); } };
    a.play().catch(() => { if (audioActual === a) audioActual = null; });
  } catch { /* sin voz */ }
}

// Las indicaciones de guía se dicen UNA sola vez mientras la app esté abierta (repetirlas cansa al niño).
const yaDichas = new Set();
export function hablarUnaVez(clave, texto, opciones) {
  if (yaDichas.has(clave)) return;
  yaDichas.add(clave);
  hablar(texto, opciones);
}
