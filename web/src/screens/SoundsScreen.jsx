import { useState } from "react";
import { SOUND_CATEGORIES } from "../soundCategories.js";
import { speakWord } from "../speak.js";
import "./LearningPathScreen.css";
import "./SoundsScreen.css";

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
          <div className="repeat-hint">Así empezás a practicar &quot;{selected.word}&quot;</div>
          <button className="sound-listen-button" onClick={() => speakWord(selected.word)}>
            🔊 Escuchar
          </button>
          <div className="repeat-note">
            Escuchá y repetí las veces que quieras. ¡Vos podés! 🌟
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
            onClick={() => {
              speakWord(item.word);
              setSelected(item);
            }}
            style={{ animationDelay: `${i * 0.05}s` }}
          >
            <span className="sound-card-speaker">🔊</span>
            <span className="sound-card-sound">{item.sound}</span>
            <span className="sound-card-icon">{item.icon}</span>
            <span className="sound-card-word">{item.word}</span>
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="path-screen">
      <div className="path-header">
        <span className="path-title">🔊 Sonidos</span>
      </div>

      {SOUND_CATEGORIES.map((cat) => (
        <div key={cat.key}>
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
