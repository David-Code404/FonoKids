// Sonidos y mensajes de refuerzo positivo (se generan con el navegador, sin archivos).

let ctx = null;
function audio() {
  try {
    ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  } catch {
    return null;
  }
}

function nota(frec, inicio, dur, vol = 0.18, tipo = "sine") {
  const c = audio();
  if (!c) return;
  const o = c.createOscillator();
  const g = c.createGain();
  o.type = tipo;
  o.frequency.value = frec;
  const t0 = c.currentTime + inicio;
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(vol, t0 + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(c.destination);
  o.start(t0);
  o.stop(t0 + dur + 0.05);
}

/// Al acertar solo cae la lluvia de estrellitas (sin sonido de aplausos).
export function sonidoBien() {
  lluviaDeEstrellas();
}

function lluviaDeEstrellas() {
  lluvia(["⭐", "🌟", "✨", "⭐"], 26, 1.6);
}
// Al fallar caen caritas tristes (más despacito y menos), en vez de estrellas.
function lluviaDeCaritas() {
  lluvia(["😢", "😞", "😟", "🥺"], 16, 2.2);
}

let estiloListo = false;
function lluvia(iconos, cuantas, minDur) {
  if (typeof document === "undefined") return;
  if (!estiloListo) {
    const s = document.createElement("style");
    s.textContent = "@keyframes fk-estrella{0%{transform:translateY(-10vh) rotate(0);opacity:0}10%{opacity:1}100%{transform:translateY(105vh) rotate(360deg);opacity:0.9}}";
    document.head.appendChild(s);
    estiloListo = true;
  }
  const capa = document.createElement("div");
  capa.style.cssText = "position:fixed;inset:0;pointer-events:none;overflow:hidden;z-index:9999";
  for (let i = 0; i < cuantas; i++) {
    const e = document.createElement("span");
    e.textContent = iconos[i % iconos.length];
    const dur = minDur + Math.random() * 1.2;
    e.style.cssText = `position:absolute;top:0;left:${Math.random() * 100}%;font-size:${22 + Math.random() * 26}px;` +
      `animation:fk-estrella ${dur}s ease-in ${Math.random() * 0.6}s forwards;opacity:0`;
    capa.appendChild(e);
  }
  document.body.appendChild(capa);
  setTimeout(() => capa.remove(), 3500);
}

/// "Uy, te falta": dos notas suaves que bajan, nada fuerte.
export function sonidoFalta() {
  lluviaDeCaritas();
  nota(392, 0, 0.25, 0.12, "triangle");
  nota(294, 0.22, 0.35, 0.12, "triangle");
}

const BIEN = ["¡Bravo!", "¡Bien hecho!", "¡Felicidades!", "¡Qué bien lo dijiste!", "¡Excelente!"];
const FALTA = ["¡Uy, te falta un poquito!", "¡Casi! Inténtalo de nuevo", "¡Tú puedes! Inténtalo de nuevo"];
const elegir = (l) => l[Math.floor(Math.random() * l.length)];

export const mensajeBien = () => `${elegir(BIEN)} ⭐`;
export const mensajeFalta = () => elegir(FALTA);
