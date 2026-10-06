import { useEffect, useRef, useState } from "react";
import { SOUND_CATEGORIES } from "../soundCategories.js";
import { fraseParaPalabra, regexPalabraEnFrase } from "../frases.js";
import { iconFor } from "../wordIcons.js";
import { getServerUrl, predictFrase } from "../api.js";
import { createAudioRecorder } from "../audioRecorder.js";
import { speakWord } from "../speak.js";
import "./LearningPathScreen.css";
import "./SoundsScreen.css";
import "./HomeScreen.css";

// Una palabra representativa por categoría -- la primera de cada una en
// soundCategories.js (mismo orden de menos a más difícil ya definido ahí).
// Con esto alcanza para el diagnóstico: no hace falta probar las 24
// palabras, con 5 (una por sonido) ya se sabe en qué categoría falla.
const PALABRAS_DIAGNOSTICO = SOUND_CATEGORIES.map((cat) => ({
  categoria: cat.key,
  titulo: cat.title,
  hint: cat.hint,
  word: cat.words[0].word,
  icon: cat.words[0].icon,
}));

const STORAGE_RESULT_KEY = "fonokids_diagnostico";
const STORAGE_DONE_KEY = "fonokids_diagnostico_done";

export function getDiagnosticoGuardado() {
  try {
    const raw = localStorage.getItem(STORAGE_RESULT_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function yaHizoDiagnostico() {
  try {
    return localStorage.getItem(STORAGE_DONE_KEY) === "1";
  } catch {
    return false;
  }
}

/// Prueba inicial: le hace decir al chico una FRASE completa por cada
/// sonido difícil (C/K, G, Sinfones, S, RR) -- no la palabra sola, pedido
/// explícito, así se acostumbra desde el principio a hablar en contexto
/// natural, igual que en la pestaña Frases. Evalúa con la misma lógica
/// combinada (GOP + clasificador entrenado sobre el audio recortado, ver
/// server/main.py POST /predict_frase) y al final arma un resumen de qué
/// sabe y qué le cuesta, para mandarlo directo a practicar con frases esas
/// palabras puntuales.
export default function DiagnosticoScreen({ onFinish }) {
  const [fase, setFase] = useState("intro"); // intro | practicando | resultado
  const [idx, setIdx] = useState(0);
  const [resultados, setResultados] = useState([]); // [{categoria, word, sabe}]
  const [micListo, setMicListo] = useState(false);
  const [micError, setMicError] = useState(null);
  const [grabando, setGrabando] = useState(false);
  const [procesando, setProcesando] = useState(false);
  const [ultimoResultado, setUltimoResultado] = useState(null);

  const streamRef = useRef(null);
  const recorderRef = useRef(createAudioRecorder());

  const actual = PALABRAS_DIAGNOSTICO[idx];

  useEffect(() => {
    if (fase !== "practicando") return;
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
  }, [fase, idx]);

  function empezar() {
    setFase("practicando");
  }

  function empezarGrabar() {
    if (!streamRef.current) return;
    const ok = recorderRef.current.start(streamRef.current);
    if (ok) {
      setGrabando(true);
      setUltimoResultado(null);
    }
  }

  async function cortarYEvaluar() {
    setGrabando(false);
    const blob = recorderRef.current.stop();
    if (!blob) return;
    setProcesando(true);
    try {
      const frase = fraseParaPalabra(actual.word);
      const r = await predictFrase(getServerUrl(), blob, frase, actual.word);
      setUltimoResultado(r);
      if (!r.valid) return; // se queda en la misma palabra, puede reintentar
      avanzar(r.veredicto_final === "bien");
    } catch (e) {
      setUltimoResultado({ valid: false, reason: e.message });
    } finally {
      setProcesando(false);
    }
  }

  function avanzar(sabe) {
    const nuevos = [...resultados, { categoria: actual.categoria, word: actual.word, sabe }];
    if (idx + 1 < PALABRAS_DIAGNOSTICO.length) {
      setResultados(nuevos);
      setIdx(idx + 1);
      setUltimoResultado(null);
    } else {
      const resumen = {};
      for (const r of nuevos) resumen[r.categoria] = r.sabe;
      try {
        localStorage.setItem(STORAGE_RESULT_KEY, JSON.stringify(resumen));
        localStorage.setItem(STORAGE_DONE_KEY, "1");
      } catch {
        // localStorage puede fallar (modo privado, cuota) -- no bloquea
        // ver el resultado igual, solo no queda guardado para la próxima.
      }
      setResultados(nuevos);
      setFase("resultado");
    }
  }

  if (fase === "intro") {
    return (
      <div className="path-screen">
        <div className="path-header">
          <span className="path-title"><span className="path-mascot">🗣️</span> Antes de arrancar...</span>
        </div>
        <div className="mode-choice-stage">
          <div className="mode-choice-title">¡Hola! Vamos a hacer una pequeña prueba</div>
          <div className="repeat-hint" style={{ maxWidth: 420, textAlign: "center" }}>
            Te voy a pedir que digas {PALABRAS_DIAGNOSTICO.length} frases completas -- así sé en
            cuáles necesitás practicar más, y te armo un camino hecho a tu medida.
          </div>
          <button className="path-cta" onClick={empezar}>
            ¡Dale, empecemos! →
          </button>
          {onFinish && (
            <button className="sound-listen-button" style={{ background: "transparent", color: "var(--text-secondary)", boxShadow: "none" }} onClick={() => onFinish(null)}>
              Prefiero saltear esto
            </button>
          )}
        </div>
      </div>
    );
  }

  if (fase === "practicando") {
    const frase = fraseParaPalabra(actual.word);
    return (
      <div className="path-screen">
        <div className="path-header">
          <span className="path-title">
            Prueba {idx + 1}/{PALABRAS_DIAGNOSTICO.length} -- {actual.titulo}
          </span>
        </div>
        <div className="repeat-stage">
          <div className="repeat-hint">Decí esta frase completa:</div>
          <div className="repeat-word" style={{ fontSize: 22, lineHeight: 1.3 }}>
            {frase.split(regexPalabraEnFrase(actual.word)).map((parte, i) =>
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

          {ultimoResultado && !ultimoResultado.valid && (
            ultimoResultado.frase_incompleta || ultimoResultado.frase_distinta ? (
              <div className="result-overlay-tip" style={{ marginTop: 8 }}>
                <div className="result-overlay-emoji">😢</div>
                <span className="result-overlay-tip-text">{ultimoResultado.reason}</span>
              </div>
            ) : (
              <div className="audio-only-error">
                {ultimoResultado.reason || "No se pudo analizar -- probá de nuevo."}
              </div>
            )
          )}
        </div>
      </div>
    );
  }

  // fase === "resultado"
  const sabeCount = resultados.filter((r) => r.sabe).length;
  return (
    <div className="path-screen">
      <div className="path-header">
        <span className="path-title"><span className="path-mascot">🗣️</span> ¡Terminamos!</span>
      </div>
      <div className="mode-choice-stage">
        <div className="mode-choice-title">
          {sabeCount === resultados.length
            ? "¡Genial, ya sabés todos estos sonidos! 🌟"
            : "Así te fue -- esto es normal, para eso practicamos"}
        </div>
        <div className="etapa-list" style={{ width: "100%", maxWidth: 480 }}>
          {resultados.map((r) => {
            const cat = PALABRAS_DIAGNOSTICO.find((p) => p.categoria === r.categoria);
            return (
              <div
                key={r.categoria}
                className={`etapa-card etapa-color-${r.sabe ? 1 : 3}`}
                style={{ cursor: "default" }}
              >
                <span className="etapa-card-word">
                  <span className="etapa-card-icon">{iconFor(cat.word)}</span> {cat.titulo}
                </span>
                <span className="etapa-card-action">
                  {r.sabe ? "¡Ya lo sabés! ✅" : "Para practicar 🔁"}
                </span>
              </div>
            );
          })}
        </div>
        <button className="path-cta" onClick={() => onFinish && onFinish(getDiagnosticoGuardado())}>
          Vamos a practicar con frases →
        </button>
      </div>
    </div>
  );
}
