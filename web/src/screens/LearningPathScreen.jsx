import { useEffect, useRef, useState } from "react";
import { getHealth, getRecordings, getServerUrl, predictFrase, wordsFromClasses } from "../api.js";
import { iconFor } from "../wordIcons.js";
import { SOUND_CATEGORIES } from "../soundCategories.js";
import { speakWord, playWordReference } from "../speak.js";
import { fraseParaPalabra, regexPalabraEnFrase } from "../frases.js";
import { createAudioRecorder } from "../audioRecorder.js";
import { tipoErrorInfo } from "../errorTips.js";
import { getFallosSeguidos, registrarIntento, resetFallos, UMBRAL_DESCANSO } from "../practiceStats.js";
import { getDiagnosticoGuardado } from "./DiagnosticoScreen.jsx";
import HomeScreen from "./HomeScreen.jsx";
import "./LearningPathScreen.css";
import "./SoundsScreen.css";

// Todas las palabras que aparecen en alguna categoría, sin duplicados --
// usado solo para calcular el contador de progreso global (cuántas ya
// tienen evaluación real vs. el total de palabras mostradas en Aprender).
const ALL_WORDS = [...new Set(SOUND_CATEGORIES.flatMap((cat) => cat.words.map((w) => w.word)))];

// A qué categoría pertenece cada palabra -- usado por el repaso espaciado
// (ver "Para repasar" abajo) para saber a dónde mandar al tocar una
// palabra que ya sabía pero hace rato no practica.
const CATEGORIA_DE_PALABRA = Object.fromEntries(
  SOUND_CATEGORIES.flatMap((cat) => cat.words.map((w) => [w.word, cat]))
);

// Repaso espaciado: una palabra que el chico YA tiene bien entra a la lista
// de "para repasar" si pasaron UMBRAL_DIAS_REPASO días o más desde la
// última vez que la practicó -- pedido explícito: que lo que ya aprendió
// no se le olvide con el tiempo, no solo mostrar palabras nuevas.
const UMBRAL_DIAS_REPASO = 3;

function diasDesde(fechaStr) {
  const dias = (Date.now() - new Date(fechaStr).getTime()) / (1000 * 60 * 60 * 24);
  return Math.floor(dias);
}

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
        ? `Decí una frase completa con "${word}" -- acá sí se evalúa de verdad.`
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
        ? `Decí una frase completa con "${displayWord}" -- acá sí se evalúa de verdad.`
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

// Un intento "cuenta" para saber si el chico YA SABE una palabra solo si
// evaluó la palabra COMPLETA (paso final con cámara/voz, class_name
// "<palabra>_correcto"/"_incorrecto") o en una frase completa (class_name
// "<palabra>_en_frase") -- los pasos intermedios de RR (sonido suelto,
// doble, a medias) no cuentan, evalúan solo un pedacito, no la palabra.
function esIntentoPalabraCompleta(row) {
  const resto = row.class_name.slice(row.word.length);
  return resto === "_en_frase" || resto === "_correcto" || resto === "_incorrecto";
}

/// Camino de aprendizaje tipo Duolingo: Etapas (una por palabra) -> Pasos
/// (4 nodos redondos por palabra) -> Práctica real (cámara + IA, o voz con
/// una frase completa) en los pasos que YA tienen datos entrenados para esa
/// familia -- los que todavía no (ver `unlockedWords`) quedan como
/// repetición libre sin evaluación, honesto en vez de fingir que califican
/// algo que no aprendieron.
///
/// Pedido explícito: unir Frases con Aprender en una sola pantalla en vez
/// de pestañas separadas -- acá, el paso final de cada palabra en modo
/// "solo voz" pide una FRASE completa (ver frases.js/predictFrase), no la
/// palabra sola; el modo cámara sigue pidiendo la palabra sola (la frase es
/// audio-only, no tiene sentido con cámara mirando la boca).
export default function LearningPathScreen() {
  const [unlockedWords, setUnlockedWords] = useState([]);
  const [view, setView] = useState("categorias"); // categorias | etapas | pasos | repetir | practicar | practicar_frase | descanso
  // A dónde ir después de la pausa de "descanso" (dificultad que se ajusta
  // sola, ver chooseMode) -- "practicar" o "practicar_frase".
  const [pendingView, setPendingView] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [selectedWord, setSelectedWord] = useState(null);
  const [selectedStepIdx, setSelectedStepIdx] = useState(0);
  // Al abrir un paso que SÍ evalúa de verdad, primero se pregunta cómo
  // quiere practicar (cámara o solo voz) -- pedido explícito: la pregunta
  // va en el momento de entrar a cada paso, no como un botón aparte.
  const [modeChoicePending, setModeChoicePending] = useState(false);
  const [audioOnlyChosen, setAudioOnlyChosen] = useState(false);

  // Pedido explícito: que Aprender también muestre lo que el chico NO sabe,
  // actualizado por cada sesión de práctica que hace -- no solo el
  // diagnóstico inicial (que se hace una sola vez). Se lee de
  // /dataset/recordings (practice_attempts en MySQL, ver server/db.py),
  // que ya guarda cada intento real con su resultado.
  const [recordings, setRecordings] = useState([]);

  async function fetchRecordings() {
    try {
      const data = await getRecordings(getServerUrl());
      const rows = data?.recordings;
      if (Array.isArray(rows)) setRecordings(rows.filter(esIntentoPalabraCompleta));
    } catch {
      // Sin conexión o sin datos todavía -- se queda con lo que ya tenía.
    }
  }

  useEffect(() => {
    fetchRecordings();
  }, []);

  // BUG real detectado: esto pedía /health UNA sola vez al montar y nunca
  // reintentaba -- si esta pantalla carga antes de que el backend termine
  // de arrancar (los modelos tardan ~15-20s en cargar), el pedido fallaba
  // esa única vez y unlockedWords quedaba en [] para siempre, mostrando
  // todo como "Practicar" (0/5) aunque el backend ya estuviera listo un
  // rato después. Ahora reintenta cada pocos segundos hasta conseguir la
  // lista, y sigue reintentando en segundo plano por si el backend se
  // reinicia mientras la app queda abierta.
  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function fetchHealth() {
      const health = await getHealth(getServerUrl());
      if (cancelled) return;
      if (health) {
        setUnlockedWords(wordsFromClasses(health.classes));
        timer = setTimeout(fetchHealth, 20000);
      } else {
        timer = setTimeout(fetchHealth, 3000);
      }
    }

    fetchHealth();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  // Última evaluación real de palabra completa (recordings ya viene
  // ordenado por fecha DESC desde el server) -- true/false/undefined si
  // todavía no hay ningún intento de esa palabra.
  function estadoPalabra(word) {
    const fila = recordings.find((r) => r.word === word);
    return fila ? fila.correcta : undefined;
  }

  // Estado de una categoría entera: prioriza la sesión más reciente de
  // cualquiera de sus palabras (más al día que el diagnóstico inicial);
  // si todavía no practicó ninguna, cae al resultado del diagnóstico.
  function estadoCategoria(cat) {
    const fila = recordings.find((r) => cat.words.some((w) => w.word === r.word));
    if (fila) return fila.correcta;
    const diagnostico = getDiagnosticoGuardado();
    return diagnostico ? diagnostico[cat.key] : undefined;
  }

  function openCategory(cat) {
    setSelectedCategory(cat);
    setView("etapas");
  }

  function openEtapa(word, cat = selectedCategory) {
    setSelectedCategory(cat);
    setSelectedWord(word);
    setSelectedStepIdx(0);
    if (cat && cat.key !== "rr") {
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

  // Repaso espaciado: salta directo a practicar una palabra que ya sabía,
  // sin pasar por la lista de categorías/etapas -- ver "Para repasar" en
  // la pantalla principal.
  function repasarPalabra(word) {
    const cat = CATEGORIA_DE_PALABRA[word];
    if (cat) openEtapa(word, cat);
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
    const steps = getSteps(selectedWord, selectedCategory, audioOnly);
    const step = steps[selectedStepIdx];
    // Frase completa solo tiene sentido en el paso final ("la palabra
    // entera") y solo en modo voz -- con cámara seguimos pidiendo la
    // palabra sola (frase es audio-only, ver predictFrase).
    const destino = audioOnly && step.key === "completa" ? "practicar_frase" : "practicar";
    // Dificultad que se ajusta sola -- pedido explícito: si viene fallando
    // seguido en esto mismo, no lo mandamos derecho a intentar de nuevo,
    // primero una pausa de repetición libre (sin evaluar), para cualquier
    // categoría, no solo RR.
    const familia = familyForStep(selectedWord, step.key);
    if (getFallosSeguidos(familia) >= UMBRAL_DESCANSO) {
      setPendingView(destino);
      setView("descanso");
    } else {
      setView(destino);
    }
  }

  function continuarDespuesDeDescanso() {
    const steps = getSteps(selectedWord, selectedCategory, audioOnlyChosen);
    const step = steps[selectedStepIdx];
    resetFallos(familyForStep(selectedWord, step.key));
    setView(pendingView || "practicar");
    setPendingView(null);
  }

  function volverDePractica() {
    const esRR = selectedCategory && selectedCategory.key === "rr";
    setView(esRR ? "pasos" : "etapas");
    fetchRecordings();
  }

  // Registra el resultado del paso actual para la dificultad adaptativa --
  // llamado tanto desde HomeScreen (cámara/voz, palabra sola) como desde
  // FrasePracticeStage (frase completa), ambos terminan en el mismo
  // contador de fallos seguidos por familia.
  function registrarResultadoPaso(correcta) {
    const steps = getSteps(selectedWord, selectedCategory, audioOnlyChosen);
    const step = steps[selectedStepIdx];
    registrarIntento(familyForStep(selectedWord, step.key), correcta);
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

  if (view === "descanso" && selectedWord) {
    const steps = getSteps(selectedWord, selectedCategory, audioOnlyChosen);
    const step = steps[selectedStepIdx];
    return (
      <div className="path-screen">
        <div className="path-header">
          <button className="icon-button" onClick={volverDePractica}>←</button>
          <span className="path-title">{iconFor(selectedWord)} {step.label}</span>
        </div>
        <div className="repeat-stage">
          <div className="result-overlay-emoji" style={{ fontSize: 48 }}>🌟</div>
          <div className="repeat-hint">
            Viniste practicando mucho esto -- ¡vamos a escuchar y repetir un ratito sin apuro antes de intentar de nuevo!
          </div>
          <div className="repeat-word">{step.label}</div>
          <button className="sound-listen-button" onClick={() => speakWord(pendingView === "practicar_frase" ? fraseParaPalabra(selectedWord) : step.label)}>
            🔊 Escuchar
          </button>
          <div className="repeat-note">
            Repetí las veces que quieras, con calma. ¡Vos podés! 🌟
          </div>
          <button className="path-cta" onClick={continuarDespuesDeDescanso}>
            Ya practiqué, ¡vamos! →
          </button>
        </div>
      </div>
    );
  }

  if (view === "practicar_frase" && selectedWord) {
    return (
      <FrasePracticeStage
        word={selectedWord}
        category={selectedCategory}
        onBack={volverDePractica}
        onResultado={registrarResultadoPaso}
      />
    );
  }

  if (view === "practicar" && selectedWord) {
    const steps = getSteps(selectedWord, selectedCategory, audioOnlyChosen);
    const step = steps[selectedStepIdx];
    return (
      <HomeScreen
        initialWord={familyForStep(selectedWord, step.key)}
        promptText={step.label}
        promptIcon={selectedWord}
        onBack={volverDePractica}
        forceAudioOnly={audioOnlyChosen}
        onResult={(result) => { if (result.valid) registrarResultadoPaso(result.correcta); }}
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
          <button className="sound-listen-button" onClick={() => playWordReference(getServerUrl(), selectedWord, displayWord)}>
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
            const resultado = estadoPalabra(w.word);
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
                  {resultado === false && <span className="etapa-card-tag" style={{ marginLeft: 8 }}>Para practicar 🔁</span>}
                  {resultado === true && <span className="etapa-card-tag" style={{ marginLeft: 8 }}>¡Ya lo sabés! ✅</span>}
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

  // Las categorías donde el chico falla (según la sesión más reciente, o
  // si todavía no practicó ninguna palabra, según el diagnóstico inicial)
  // van primero -- pedido explícito: que el camino priorice lo que le
  // cuesta al chico en vez de mostrar las 5 por igual, y que se actualice
  // sesión a sesión en vez de quedar fijo con el diagnóstico de una vez.
  const categoriasOrdenadas = [...SOUND_CATEGORIES].sort((a, b) => {
    const aFalla = estadoCategoria(a) === false;
    const bFalla = estadoCategoria(b) === false;
    if (aFalla === bFalla) return 0;
    return aFalla ? -1 : 1;
  });

  // Repaso espaciado: palabras que ya salieron bien pero hace
  // UMBRAL_DIAS_REPASO días o más que no se practican -- una por palabra
  // (recordings ya viene ordenado por fecha DESC, el primer match es el
  // más reciente).
  const vistos = new Set();
  const paraRepasar = [];
  for (const r of recordings) {
    if (vistos.has(r.word)) continue;
    vistos.add(r.word);
    if (r.correcta && diasDesde(r.date) >= UMBRAL_DIAS_REPASO) {
      paraRepasar.push({ word: r.word, dias: diasDesde(r.date) });
    }
  }
  paraRepasar.sort((a, b) => b.dias - a.dias);

  return (
    <div className="path-screen">
      <div className="path-header">
        <span className="path-title"><span className="path-mascot">🗣️</span> FonoKids -- Camino</span>
        <span className="path-progress-pill">
          🏆 {unlockedWords.length}/{ALL_WORDS.length} etapas
        </span>
      </div>

      {paraRepasar.length > 0 && (
        <div className="sounds-category">
          <span className="sounds-category-title">🔄 Para repasar</span>
          <span className="sounds-category-hint">
            Ya las sabías -- practiquemos de nuevo para que no se te olviden
          </span>
        </div>
      )}
      {paraRepasar.length > 0 && (
        <div className="etapa-list">
          {paraRepasar.map(({ word, dias }, i) => {
            const cat = CATEGORIA_DE_PALABRA[word];
            const wordInfo = cat?.words.find((w) => w.word === word);
            return (
              <button
                key={word}
                className={`etapa-card etapa-color-${i % 4}`}
                onClick={() => repasarPalabra(word)}
                style={{ animationDelay: `${i * 0.06}s` }}
              >
                <span className="etapa-card-word">
                  <span className="etapa-card-icon">{wordInfo?.icon || iconFor(word)}</span> {(wordInfo?.label || word).replaceAll("_", " ")}
                </span>
                <span className="etapa-card-action">
                  Hace {dias} {dias === 1 ? "día" : "días"} →
                </span>
              </button>
            );
          })}
        </div>
      )}

      <div className="etapa-list">
        {categoriasOrdenadas.map((cat, i) => {
          const resultado = estadoCategoria(cat);
          return (
            <button
              key={cat.key}
              className={`etapa-card etapa-color-${i % 4}`}
              onClick={() => openCategory(cat)}
              style={{ animationDelay: `${i * 0.06}s` }}
            >
              <span className="etapa-card-word">
                <span className="etapa-card-icon">{cat.words[0]?.icon}</span> {cat.title}
                {resultado === false && <span className="etapa-card-tag" style={{ marginLeft: 8 }}>Para practicar</span>}
                {resultado === true && <span className="etapa-card-tag" style={{ marginLeft: 8 }}>¡Ya lo sabés!</span>}
              </span>
              <span className="etapa-card-action">
                {cat.hint}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// Práctica con FRASE completa para el paso final de una palabra en modo
// "solo voz" -- fusiona lo que antes era la pestaña "Frases" aparte, ahora
// vive directamente dentro del paso final de Aprender (pedido explícito:
// una sola pantalla en vez de dos). Misma lógica que server/main.py
// POST /predict_frase (GOP + clasificador entrenado sobre el audio
// recortado).
function FrasePracticeStage({ word, category, onBack, onResultado }) {
  const [micListo, setMicListo] = useState(false);
  const [micError, setMicError] = useState(null);
  const [grabando, setGrabando] = useState(false);
  const [procesando, setProcesando] = useState(false);
  const [resultado, setResultado] = useState(null);

  const streamRef = useRef(null);
  const recorderRef = useRef(createAudioRecorder());

  const wordInfo = category?.words.find((w) => w.word === word);
  const displayWord = wordInfo?.label || word;
  const frase = fraseParaPalabra(word);

  useEffect(() => {
    let cancelled = false;
    setMicError(null);
    setMicListo(false);
    navigator.mediaDevices
      ?.getUserMedia({ audio: true })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        setMicListo(true);
      })
      .catch((e) => {
        if (!cancelled) setMicError(e.message || "No se pudo prender el micrófono.");
      });
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [word]);

  function empezarGrabar() {
    if (!streamRef.current) return;
    const ok = recorderRef.current.start(streamRef.current);
    if (ok) {
      setGrabando(true);
      setResultado(null);
    }
  }

  async function cortarYEvaluar() {
    setGrabando(false);
    const blob = recorderRef.current.stop();
    if (!blob) return;
    setProcesando(true);
    try {
      const r = await predictFrase(getServerUrl(), blob, frase, word);
      setResultado(r);
      if (r.valid && onResultado) onResultado(r.correcta);
    } catch (e) {
      setResultado({ valid: false, reason: e.message || "Error al analizar el audio." });
    } finally {
      setProcesando(false);
    }
  }

  const errorInfo = resultado?.valid && resultado.tipo_error ? tipoErrorInfo(resultado.tipo_error) : null;

  return (
    <div className="path-screen">
      <div className="path-header">
        <button className="icon-button" onClick={onBack}>←</button>
        <span className="path-title">{wordInfo?.icon || iconFor(word)} {displayWord}</span>
      </div>
      <div className="repeat-stage">
        <div className="repeat-hint">Decí esta frase completa:</div>
        <div className="repeat-word" style={{ fontSize: 22, lineHeight: 1.3 }}>
          {frase.split(regexPalabraEnFrase(word)).map((parte, i) =>
            i % 2 === 1
              ? <span key={i} style={{ color: "var(--accent, #6C5CE7)", textDecoration: "underline" }}>{parte}</span>
              : <span key={i}>{parte}</span>
          )}
        </div>
        <button className="sound-listen-button" onClick={() => speakWord(frase)}>
          🔊 Escuchar
        </button>

        {micError ? (
          <div className="audio-only-error">No se pudo usar el micrófono: {micError}</div>
        ) : !micListo ? (
          <div className="audio-only-label">Activando el micrófono...</div>
        ) : (
          <div className="audio-only-stage" style={{ marginTop: 8 }}>
            <div className={`audio-only-orb ${grabando ? "is-recording" : ""}`}>
              {grabando && (
                <>
                  <span className="audio-only-pulse audio-only-pulse-1" />
                  <span className="audio-only-pulse audio-only-pulse-2" />
                </>
              )}
              <span className="audio-only-icon">🎤</span>
            </div>
            <span className="audio-only-label">
              {procesando ? "Analizando..." : grabando ? "Escuchando..." : "Tocá para grabar"}
            </span>
          </div>
        )}

        {micListo && !procesando && (
          <button className="path-cta" onClick={grabando ? cortarYEvaluar : empezarGrabar}>
            {grabando ? "Listo, evaluar →" : "🎤 Grabar"}
          </button>
        )}

        {resultado && (
          resultado.valid ? (
            <div className="result-overlay-tip" style={{ marginTop: 8 }}>
              <div className="result-overlay-emoji">{resultado.correcta ? "😄" : "😢"}</div>
              <span className="result-overlay-tip-label">
                {resultado.correcta ? "¡Muy bien dicho!" : (errorInfo?.label || "¡Casi! Practiquemos de nuevo")}
              </span>
              {!resultado.correcta && (
                <>
                  <span className="result-overlay-tip-text">
                    {errorInfo?.tip || "Probemos de nuevo, despacito. ¡Vos podés!"}
                  </span>
                  <button className="path-cta" style={{ marginTop: 10 }} onClick={empezarGrabar}>
                    🎤 Intentar de nuevo
                  </button>
                </>
              )}
            </div>
          ) : resultado.frase_incompleta ? (
            <div className="result-overlay-tip" style={{ marginTop: 8 }}>
              <div className="result-overlay-emoji">😅</div>
              <span className="result-overlay-tip-text">
                Decí toda la frase, despacito -- ¡vos podés!
              </span>
              <button className="path-cta" style={{ marginTop: 10 }} onClick={empezarGrabar}>
                🎤 Intentar de nuevo
              </button>
            </div>
          ) : resultado.frase_distinta ? (
            <div className="result-overlay-tip" style={{ marginTop: 8 }}>
              <div className="result-overlay-emoji">😅</div>
              <span className="result-overlay-tip-text">
                Repetí la frase de arriba tal cual está -- ¡dale, de nuevo!
              </span>
              <button className="path-cta" style={{ marginTop: 10 }} onClick={empezarGrabar}>
                🎤 Intentar de nuevo
              </button>
            </div>
          ) : (
            <div className="audio-only-error">{resultado.reason || "No se pudo analizar."}</div>
          )
        )}
      </div>
    </div>
  );
}
