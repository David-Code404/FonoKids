import { useEffect, useState } from "react";
import { getHealth, getServerUrl, wordsFromClasses } from "../api.js";
import { iconFor } from "../wordIcons.js";
import { SOUND_CATEGORIES } from "../soundCategories.js";
import { speakWord } from "../speak.js";
import HomeScreen from "./HomeScreen.jsx";
import "./LearningPathScreen.css";
import "./SoundsScreen.css";

// Todas las palabras que aparecen en alguna categoría, sin duplicados --
// usado solo para calcular el contador de progreso global (cuántas ya
// tienen evaluación real vs. el total de palabras mostradas en Aprender).
const ALL_WORDS = [...new Set(SOUND_CATEGORIES.flatMap((cat) => cat.words.map((w) => w.word)))];

// Desglose en 4 pasos (sonido suelto -> doble -> palabra a medias ->
// palabra completa) -- MISMO patrón para cualquier palabra, a propósito
// simplificado (no es un análisis fonético fino, es un andamiaje
// pedagógico visual: practicar el sonido difícil suelto, reforzarlo, y
// después ir agrandando hasta la palabra entera).
function buildSteps(word, audioOnly = false) {
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
      desc: audioOnly
        ? `Decí "${word}" completa -- acá sí se evalúa de verdad.`
        : `Decí "${word}" completa, con cámara -- acá sí se evalúa de verdad.`,
    },
  ];
}

// Para categorías fuera de RR (S, Sinfones, G suave, C/K), el sonido difícil
// está al PRINCIPIO de la palabra, no en el medio -- no existe el desglose
// de "a medias"/"doble" (eso es exclusivo de RR). Esas categorías ya tienen
// clases entrenadas de la palabra completa (correcto/incorrecto), así que
// alcanza con un solo paso evaluable en vez de los 4 de buildSteps().
function getSteps(word, category, audioOnly = false) {
  if (category && category.key === "rr") return buildSteps(word, audioOnly);
  const displayWord = category?.words.find((w) => w.word === word)?.label || word;
  return [
    {
      key: "completa", label: displayWord.toUpperCase(), hint: "¡Decilo completo!",
      desc: audioOnly
        ? `Decí "${displayWord}" completa -- acá sí se evalúa de verdad.`
        : `Decí "${displayWord}" completa, con cámara -- acá sí se evalúa de verdad.`,
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
  const [view, setView] = useState("categorias"); // categorias | etapas | pasos | repetir | practicar
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [selectedWord, setSelectedWord] = useState(null);
  const [selectedStepIdx, setSelectedStepIdx] = useState(0);
  // Al abrir un paso que SÍ evalúa de verdad, primero se pregunta cómo
  // quiere practicar (cámara o solo voz) -- pedido explícito: la pregunta
  // va en el momento de entrar a cada paso, no como un botón aparte.
  const [modeChoicePending, setModeChoicePending] = useState(false);
  const [audioOnlyChosen, setAudioOnlyChosen] = useState(false);

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

  function openCategory(cat) {
    setSelectedCategory(cat);
    setView("etapas");
  }

  function openEtapa(word) {
    setSelectedWord(word);
    setSelectedStepIdx(0);
    if (selectedCategory && selectedCategory.key !== "rr") {
      // Categorías fuera de RR (S, Sinfones, G suave, C/K) no tienen el
      // desglose de 4 pasos (eso es exclusivo de RR) -- si la palabra ya
      // tiene clases entrenadas (ver unlockedWords), va directo a elegir
      // cámara/voz y evaluar de verdad; si no, queda como repetición libre.
      if (unlockedWords.includes(word)) {
        setModeChoicePending(true);
      } else {
        setView("detalle");
      }
      return;
    }
    // Todas las etapas de RR quedan abiertas (pedido explícito) -- las que
    // todavía no tienen datos entrenados igual dejan repasar los sonidos
    // sueltos (pasos 1-3, que no necesitan modelo), solo el paso final
    // (palabra completa con cámara) va a avisar si esa palabra no tiene
    // evaluación real disponible todavía.
    setView("pasos");
  }

  function openPaso(idx) {
    setSelectedStepIdx(idx);
    const steps = buildSteps(selectedWord);
    const familia = familyForStep(selectedWord, steps[idx].key);
    if (unlockedWords.includes(familia)) {
      // Este paso SÍ evalúa de verdad -- antes de entrar, preguntamos cómo
      // quiere practicar (cámara o solo voz).
      setModeChoicePending(true);
    } else {
      setView("repetir");
    }
  }

  function chooseMode(audioOnly) {
    setAudioOnlyChosen(audioOnly);
    setModeChoicePending(false);
    setView("practicar");
  }

  if (modeChoicePending && selectedWord) {
    const steps = getSteps(selectedWord, selectedCategory);
    const step = steps[selectedStepIdx];
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={() => setModeChoicePending(false)}>←</button>
          <span className="path-title">{iconFor(selectedWord)} {step.label}</span>
        </div>
        <div className="mode-choice-stage">
          <div className="mode-choice-title">¿Cómo querés practicar?</div>
          <div className="mode-choice-buttons">
            <button className="mode-choice-button" onClick={() => chooseMode(false)}>
              <span className="mode-choice-icon">📷</span>
              Con cámara
            </button>
            <button className="mode-choice-button" onClick={() => chooseMode(true)}>
              <span className="mode-choice-icon">🎤</span>
              Solo con la voz
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (view === "practicar" && selectedWord) {
    const steps = getSteps(selectedWord, selectedCategory, audioOnlyChosen);
    const step = steps[selectedStepIdx];
    const esRR = selectedCategory && selectedCategory.key === "rr";
    return (
      <HomeScreen
        initialWord={familyForStep(selectedWord, step.key)}
        promptText={step.label}
        promptIcon={selectedWord}
        onBack={() => setView(esRR ? "pasos" : "etapas")}
        forceAudioOnly={audioOnlyChosen}
      />
    );
  }

  if (view === "detalle" && selectedWord) {
    const wordInfo = selectedCategory?.words.find((w) => w.word === selectedWord);
    const displayWord = wordInfo?.label || selectedWord;
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={() => setView("etapas")}>←</button>
          <span className="path-title">{wordInfo?.icon || iconFor(selectedWord)} {displayWord}</span>
        </div>
        <div className="repeat-stage">
          <div className="repeat-word">{wordInfo?.sound}</div>
          <div className="repeat-hint">Así empezás a practicar &quot;{displayWord}&quot;</div>
          <button className="sound-listen-button" onClick={() => speakWord(displayWord)}>
            🔊 Escuchar
          </button>
          <div className="repeat-note">
            Escuchá y repetí las veces que quieras. ¡Vos podés! 🌟
          </div>
        </div>
      </div>
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
            Practicá en voz alta las veces que quieras antes de pasar al siguiente paso.
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

  if (view === "etapas" && selectedCategory) {
    const words = selectedCategory.words;
    const unlockedHere = words.filter((w) => unlockedWords.includes(w.word)).length;
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={() => setView("categorias")}>←</button>
          <span className="path-title">{selectedCategory.title}</span>
          <span className="path-progress-pill">
            🏆 {unlockedHere}/{words.length}
          </span>
        </div>
        <div className="etapa-list">
          {words.map((w, i) => {
            const unlocked = unlockedWords.includes(w.word);
            return (
              <button
                key={w.word}
                className={`etapa-card etapa-color-${i % 4}`}
                onClick={() => openEtapa(w.word)}
                style={{ animationDelay: `${i * 0.06}s` }}
              >
                <span className="etapa-card-tag">Etapa {i + 1}</span>
                <span className="etapa-card-word">
                  <span className="etapa-card-icon">{w.icon || iconFor(w.word)}</span> {(w.label || w.word).replaceAll("_", " ")}
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

  return (
    <div className="path-screen">
      <div className="path-header">
        <span className="path-title"><span className="path-mascot">🗣️</span> FonoKids -- Camino</span>
        <span className="path-progress-pill">
          🏆 {unlockedWords.length}/{ALL_WORDS.length} etapas
        </span>
      </div>
      <div className="etapa-list">
        {SOUND_CATEGORIES.map((cat, i) => (
          <button
            key={cat.key}
            className={`etapa-card etapa-color-${i % 4}`}
            onClick={() => openCategory(cat)}
            style={{ animationDelay: `${i * 0.06}s` }}
          >
            <span className="etapa-card-word">
              <span className="etapa-card-icon">{cat.words[0]?.icon}</span> {cat.title}
            </span>
            <span className="etapa-card-action">
              {cat.hint}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
