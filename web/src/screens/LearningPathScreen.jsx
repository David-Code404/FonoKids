import { useEffect, useState } from "react";
import { getHealth, getServerUrl, wordsFromClasses } from "../api.js";
import { iconFor } from "../wordIcons.js";
import HomeScreen from "./HomeScreen.jsx";
import "./LearningPathScreen.css";

// Las 9 palabras objetivo del proyecto, en el orden fijo que define el
// número de "Etapa" -- no viene del server (que solo lista las que YA
// tienen clases entrenadas), esta es la lista completa del plan, así se
// pueden ver las etapas bloqueadas (sin grabar/entrenar todavía) también.
const ALL_WORDS = ["perro", "carro", "burro", "gorra", "torre"];

// Desglose en 4 pasos (sonido suelto -> doble -> palabra a medias ->
// palabra completa) -- MISMO patrón para cualquier palabra, a propósito
// simplificado (no es un análisis fonético fino, es un andamiaje
// pedagógico visual: practicar el sonido difícil suelto, reforzarlo, y
// después ir agrandando hasta la palabra entera).
function buildSteps(word) {
  const hasDoubleR = word.includes("rr");
  const partial = word.length > 4 ? word.slice(0, -2) : word.slice(0, -1);
  return [
    {
      key: "sonido", label: "r", hint: "Practicá el sonido solo",
      desc: "Un solo golpecito de lengua contra el paladar, suave -- sin vibrar.",
    },
    {
      key: "doble", label: hasDoubleR ? "rr" : "r", hint: "Practicá el sonido fuerte",
      desc: "La lengua vibra varias veces seguidas -- un sonido fuerte, como un motorcito.",
    },
    {
      key: "medias", label: partial.toUpperCase(), hint: "Ya casi la palabra entera",
      desc: `Decí "${partial}" con la R bien pronunciada al final.`,
    },
    {
      key: "completa", label: word.toUpperCase(), hint: "¡La palabra completa!",
      desc: `Decí "${word}" completa, con cámara -- acá sí se evalúa de verdad.`,
    },
  ];
}

// Nombre de la FAMILIA de clases que le corresponde a un paso -- "completa"
// usa la palabra tal cual ("perro"), los demás pasos son sub-familias
// ("perro_doble", "perro_medias", "perro_sonido") -- mismo criterio que
// server/main.py (_classify_audio filtra clases por este prefijo, y ahora
// también rechaza un resultado si la clase ganadora es de OTRA familia).
function familyForStep(word, stepKey) {
  return stepKey === "completa" ? word : `${word}_${stepKey}`;
}

/// Camino de aprendizaje tipo Duolingo: Etapas (una por palabra) -> Pasos
/// (4 nodos redondos por palabra) -> Práctica real (cámara + IA) en los
/// pasos que YA tienen datos entrenados para esa familia -- los que
/// todavía no (ver `unlockedWords`) quedan como repetición libre sin
/// evaluación, honesto en vez de fingir que califican algo que no aprendieron.
export default function LearningPathScreen() {
  const [unlockedWords, setUnlockedWords] = useState([]);
  const [view, setView] = useState("etapas"); // etapas | pasos | repetir | practicar
  const [selectedWord, setSelectedWord] = useState(null);
  const [selectedStepIdx, setSelectedStepIdx] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const health = await getHealth(getServerUrl());
      if (cancelled || !health) return;
      setUnlockedWords(wordsFromClasses(health.classes));
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  function openEtapa(word) {
    // Todas las etapas quedan abiertas (pedido explícito) -- las que
    // todavía no tienen datos entrenados igual dejan repasar los sonidos
    // sueltos (pasos 1-3, que no necesitan modelo), solo el paso final
    // (palabra completa con cámara) va a avisar si esa palabra no tiene
    // evaluación real disponible todavía.
    setSelectedWord(word);
    setView("pasos");
  }

  function openPaso(idx) {
    setSelectedStepIdx(idx);
    const steps = buildSteps(selectedWord);
    const familia = familyForStep(selectedWord, steps[idx].key);
    setView(unlockedWords.includes(familia) ? "practicar" : "repetir");
  }

  if (view === "practicar" && selectedWord) {
    const steps = buildSteps(selectedWord);
    const step = steps[selectedStepIdx];
    return (
      <HomeScreen
        initialWord={familyForStep(selectedWord, step.key)}
        promptText={step.label}
        promptIcon={selectedWord}
        onBack={() => setView("pasos")}
      />
    );
  }

  if (view === "repetir" && selectedWord) {
    const steps = buildSteps(selectedWord);
    const step = steps[selectedStepIdx];
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={() => setView("pasos")}>←</button>
          <span className="path-title">{selectedWord.replaceAll("_", " ")}</span>
        </div>
        <div className="repeat-stage">
          <div className="repeat-word">{step.label}</div>
          <div className="repeat-hint">{step.hint}</div>
          <div className="repeat-note">
            Repetí en voz alta las veces que quieras -- todavía no grabamos suficientes
            ejemplos de este paso para evaluarlo con IA, así que por ahora es solo práctica
            libre (algunos de los otros pasos ya sí evalúan con cámara).
          </div>
          <button className="path-cta" onClick={() => openPaso(selectedStepIdx + 1)}>
            Listo, seguir →
          </button>
        </div>
      </div>
    );
  }

  if (view === "pasos" && selectedWord) {
    const steps = buildSteps(selectedWord);
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={() => setView("etapas")}>←</button>
          <span className="path-title">{iconFor(selectedWord)} {selectedWord.replaceAll("_", " ")}</span>
        </div>
        <div className="paso-list">
          {steps.map((step, i) => (
            <button key={step.key} className={`paso-card step-${i}`} onClick={() => openPaso(i)}>
              <span className="paso-card-badge">{step.label}</span>
              <span className="paso-card-body">
                <span className="paso-card-hint">{step.hint}</span>
                <span className="paso-card-desc">{step.desc}</span>
              </span>
              <span className="paso-card-arrow">→</span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="path-screen">
      <div className="path-header">
        <span className="path-title"><span className="path-mascot">🗣️</span> FonoKids -- Camino</span>
        <span className="path-progress-pill">
          🏆 {unlockedWords.length}/{ALL_WORDS.length} etapas
        </span>
      </div>
      <div className="etapa-list">
        {ALL_WORDS.map((word, i) => {
          const unlocked = unlockedWords.includes(word);
          return (
            <button
              key={word}
              className={`etapa-card etapa-color-${i % 4}`}
              onClick={() => openEtapa(word)}
              style={{ animationDelay: `${i * 0.06}s` }}
            >
              <span className="etapa-card-tag">Etapa {i + 1}</span>
              <span className="etapa-card-word">
                <span className="etapa-card-icon">{iconFor(word)}</span> {word.replaceAll("_", " ")}
              </span>
              <span className="etapa-card-action">
                {unlocked ? "Repasar →" : "Practicar →"}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
