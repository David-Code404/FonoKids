import { useState } from "react";
import { iconFor } from "../wordIcons.js";
import "./LearningPathScreen.css";
import "./SoundsScreen.css";

// Las 9 palabras objetivo del proyecto -- mismo orden que en
// LearningPathScreen.jsx (Etapas). Son las únicas con datos entrenados
// (por ahora solo "perro" tiene evaluación real en Aprender), pero acá en
// Sonidos se muestran igual como referencia de "cómo empezar a practicar".
const ALL_WORDS = [
  "perro", "carro", "raton", "mariposa", "tren",
  "fresa", "rueda", "sombrero", "bicicleta",
];

// El sonido clave para arrancar a practicar cada palabra -- todas trabajan
// la r, "rr" si la palabra la tiene escrita doble, "r" si no (que igual se
// pronuncia fuerte/vibrante al principio de palabra o después de n/l/s,
// mismo criterio que texto_a_fonemas() en audio_pronunciation.py).
function keySoundFor(word) {
  return word.includes("rr") ? "rr" : "r";
}

// El resto de los sonidos difíciles del español (dislalias más comunes en
// chicos) -- pedido explícito para mostrarlos TODOS en Sonidos, aunque hoy
// ninguno de estos tenga clases entrenadas todavía (eso queda para más
// adelante, por ahora solo "perro" evalúa de verdad en Aprender). Son
// palabras de referencia/práctica libre, igual criterio honesto que ya
// usan R/RR acá: repetir sin evaluación real.
const OTHER_CATEGORIES = [
  {
    title: "Sinfones con R",
    hint: "dos consonantes seguidas, la boca cambia rápido de posición",
    words: [
      { word: "tren", sound: "tr", icon: "🚂" },
      { word: "fresa", sound: "fr", icon: "🍓" },
      { word: "brazo", sound: "br", icon: "💪" },
      { word: "cruz", sound: "cr", icon: "✝️" },
      { word: "dragón", sound: "dr", icon: "🐉" },
      { word: "grillo", sound: "gr", icon: "🦗" },
      { word: "premio", sound: "pr", icon: "🎁" },
    ],
  },
  {
    title: "Sinfones con L",
    hint: "otra combinación de dos consonantes que cuesta coordinar",
    words: [
      { word: "blanco", sound: "bl", icon: "⬜" },
      { word: "clase", sound: "cl", icon: "🏫" },
      { word: "flor", sound: "fl", icon: "🌸" },
      { word: "globo", sound: "gl", icon: "🎈" },
      { word: "plátano", sound: "pl", icon: "🍌" },
    ],
  },
  {
    title: "S (sigmatismo)",
    hint: "la lengua se va entre los dientes y suena como una Z",
    words: [
      { word: "sol", sound: "s", icon: "☀️" },
      { word: "sapo", sound: "s", icon: "🐸" },
    ],
  },
  {
    title: "D / Z",
    hint: "la lengua no llega bien a la punta de los dientes de arriba",
    words: [
      { word: "dado", sound: "d", icon: "🎲" },
      { word: "zapato", sound: "z", icon: "👟" },
    ],
  },
  {
    title: "CH / LL-Y",
    hint: "necesitan que la lengua presione fuerte contra el paladar",
    words: [
      { word: "chocolate", sound: "ch", icon: "🍫" },
      { word: "lluvia", sound: "ll", icon: "🌧️" },
    ],
  },
  {
    title: "J / G fuerte",
    hint: "se raspa el aire en la parte de atrás de la garganta",
    words: [
      { word: "jirafa", sound: "j", icon: "🦒" },
      { word: "ojo", sound: "j", icon: "👁️" },
    ],
  },
  {
    title: "K / G suave",
    hint: "sonidos de atrás de la boca -- a veces se cambian por T/D",
    words: [
      { word: "casa", sound: "k", icon: "🏠" },
      { word: "agua", sound: "g", icon: "💧" },
    ],
  },
  {
    title: "F",
    hint: "el labio de abajo tiene que apoyarse en los dientes de arriba",
    words: [
      { word: "foco", sound: "f", icon: "💡" },
    ],
  },
  {
    title: "Consonantes al final",
    hint: "las últimas letras de la palabra se comen o cambian",
    words: [
      { word: "reloj", sound: "finales", icon: "⏰" },
      { word: "actor", sound: "finales", icon: "🎭" },
    ],
  },
];

export default function SoundsScreen() {
  const [selected, setSelected] = useState(null);

  if (selected) {
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={() => setSelected(null)}>←</button>
          <span className="path-title">{selected.icon} {selected.word}</span>
        </div>
        <div className="repeat-stage">
          <div className="repeat-word">{selected.sound}</div>
          <div className="repeat-hint">Así empezás a practicar "{selected.word}"</div>
          <div className="repeat-note">
            Repetí en voz alta las veces que quieras -- esto es solo para practicar,
            no evalúa (hoy la evaluación real con cámara solo existe para "perro",
            en "Aprender"; las demás palabras se van a ir sumando más adelante).
          </div>
        </div>
      </div>
    );
  }

  function renderGrid(words) {
    return (
      <div className="sounds-grid">
        {words.map((item, i) => (
          <button
            key={item.word}
            className={`sound-card etapa-color-${i % 4}`}
            onClick={() => setSelected(item)}
            style={{ animationDelay: `${i * 0.05}s` }}
          >
            <span className="sound-card-sound">{item.sound}</span>
            <span className="sound-card-icon">{item.icon}</span>
            <span className="sound-card-word">{item.word}</span>
          </button>
        ))}
      </div>
    );
  }

  const rSuave = ALL_WORDS.filter((w) => keySoundFor(w) === "r")
    .map((w) => ({ word: w.replaceAll("_", " "), sound: "r", icon: iconFor(w) }));
  const rFuerte = ALL_WORDS.filter((w) => keySoundFor(w) === "rr")
    .map((w) => ({ word: w.replaceAll("_", " "), sound: "rr", icon: iconFor(w) }));

  return (
    <div className="path-screen">
      <div className="path-header">
        <span className="path-title">🔊 Sonidos</span>
      </div>

      <div className="sounds-category">
        <span className="sounds-category-title">R suave</span>
        <span className="sounds-category-hint">un solo golpecito de lengua, sin vibrar</span>
      </div>
      {renderGrid(rSuave)}

      <div className="sounds-category">
        <span className="sounds-category-title">RR fuerte</span>
        <span className="sounds-category-hint">lengua vibrando, sonido bien marcado</span>
      </div>
      {renderGrid(rFuerte)}

      {OTHER_CATEGORIES.map((cat) => (
        <div key={cat.title}>
          <div className="sounds-category">
            <span className="sounds-category-title">{cat.title}</span>
            <span className="sounds-category-hint">{cat.hint}</span>
          </div>
          {renderGrid(cat.words)}
        </div>
      ))}
    </div>
  );
}
