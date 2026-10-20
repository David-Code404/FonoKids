import { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { getServerUrl, ejercicioVoz } from "../api.js";
import { createAudioRecorder } from "../audioRecorder.js";
import { EJERCICIOS, ETIQUETA_TIPO, HOLD_MS, REPETICIONES } from "../ejercicios.js";
import { mostrarPremio, registrarIntentoValido, textoPremio } from "../rachas.js";
import "./LearningPathScreen.css";
import "./SoundsScreen.css";
import "./HomeScreen.css";
import "./EjerciciosScreen.css";

// Mismo modelo y mismo origen que usa HomeScreen para el overlay de puntos.
const MEDIAPIPE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const MEDIAPIPE_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const CALIBRACION_MS = 2000;

/// Premio por completar un ejercicio: cuenta como práctica del día (racha)
/// y da una estrella, igual que un intento bien dicho en Aprender.
function darPremio() {
  const premio = textoPremio(registrarIntentoValido(true));
  mostrarPremio(premio || "⭐ ¡Muy bien!");
}

function distancia(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/// Señales del gesto (lista de valores), a partir de los blendshapes de
/// MediaPipe y, para las mejillas, también de los puntos de la cara: el
/// blendshape `cheekPuff` suele dar valores muy bajos, así que además se mide
/// cuánto se ensanchan las mejillas (puntos 205 y 425) respecto a la
/// distancia entre los ojos (33 y 263), que no cambia al inflarlas.
function medir(categorias, cara, medida) {
  const v = (nombre) => categorias.find((c) => c.categoryName === nombre)?.score ?? 0;
  switch (medida) {
    case "sonrisa": return [(v("mouthSmileLeft") + v("mouthSmileRight")) / 2];
    case "beso": return [v("mouthPucker")];
    case "boca": return [v("jawOpen")];
    case "mejillas": {
      let ancho = 0;
      if (cara?.[205] && cara?.[425] && cara?.[33] && cara?.[263]) {
        ancho = distancia(cara[205], cara[425]) / Math.max(distancia(cara[33], cara[263]), 1e-6);
      }
      return [v("cheekPuff"), ancho];
    }
    default: return [0];
  }
}

function Encabezado({ icono, titulo, onBack }) {
  return (
    <div className="path-header">
      <button className="icon-button" onClick={onBack}>←</button>
      <span className="path-title">{icono} {titulo}</span>
    </div>
  );
}

// ---------------------------------------------------------------- PRAXIA
function PraxiaStage({ ejercicio, onBack }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const faseRef = useRef("cargando");
  const calRef = useRef({ t0: 0, suma: [], n: 0, base: [] });
  const holdRef = useRef({ inicio: null, necesitaSoltar: false, reps: 0 });

  const [fase, setFase] = useState("cargando"); // cargando | calibrando | haciendo | listo | error
  const [error, setError] = useState(null);
  const [progreso, setProgreso] = useState(0);
  const [reps, setReps] = useState(0);
  const [hayCara, setHayCara] = useState(true);
  const [senal, setSenal] = useState(0); // 0..1: cuánto falta para que cuente el gesto

  function cambiarFase(f) {
    faseRef.current = f;
    setFase(f);
  }

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: 640, height: 480 }, audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }

        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
        const landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MEDIAPIPE_MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
        });
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;
        calRef.current = { t0: performance.now(), suma: [], n: 0, base: [] };
        cambiarFase("calibrando");
        rafRef.current = requestAnimationFrame(tick);
      } catch (e) {
        if (!cancelled) {
          setError(e?.name === "NotAllowedError"
            ? "Necesito permiso para usar la cámara."
            : "No pude abrir la cámara o cargar el detector de la cara (¿hay internet?).");
          cambiarFase("error");
        }
      }
    })();

    function tick() {
      rafRef.current = requestAnimationFrame(tick);
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;
      if (!video || !landmarker || video.readyState < 2) return;
      if (faseRef.current !== "calibrando" && faseRef.current !== "haciendo") return;

      const ahora = performance.now();
      const res = landmarker.detectForVideo(video, ahora);
      const cats = res.faceBlendshapes?.[0]?.categories;
      if (!cats) {
        setHayCara(false);
        holdRef.current.inicio = null;
        setProgreso(0);
        return;
      }
      setHayCara(true);
      const valores = medir(cats, res.faceLandmarks?.[0], ejercicio.medida);

      if (faseRef.current === "calibrando") {
        const c = calRef.current;
        c.suma = valores.map((x, i) => (c.suma[i] ?? 0) + x);
        c.n += 1;
        if (ahora - c.t0 >= CALIBRACION_MS && c.n > 5) {
          c.base = c.suma.map((x) => x / c.n);
          cambiarFase("haciendo");
        }
        return;
      }

      // Señal del gesto: 1 = justo lo necesario. Con varias medidas (mejillas)
      // alcanza con que una de ellas llegue.
      const deltas = ejercicio.deltas;
      const senalActual = Math.max(...valores.map((x, i) => (x - (calRef.current.base[i] ?? 0)) / deltas[i]));
      setSenal(Math.max(0, Math.min(senalActual, 1)));

      const h = holdRef.current;
      const sobre = senalActual >= 1;
      if (!sobre) {
        h.inicio = null;
        h.necesitaSoltar = false;
        setProgreso(0);
        return;
      }
      if (h.necesitaSoltar) return; // ya contó esta repetición: primero tiene que relajar la cara
      if (h.inicio === null) h.inicio = ahora;
      const prog = Math.min((ahora - h.inicio) / HOLD_MS, 1);
      setProgreso(prog);
      if (prog >= 1) {
        h.reps += 1;
        h.inicio = null;
        h.necesitaSoltar = true;
        setReps(h.reps);
        setProgreso(0);
        if (h.reps >= REPETICIONES) {
          cambiarFase("listo");
          darPremio();
        }
      }
    }

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ejercicio.id]);

  function repetir() {
    holdRef.current = { inicio: null, necesitaSoltar: false, reps: 0 };
    setReps(0);
    setProgreso(0);
    calRef.current = { t0: performance.now(), suma: [], n: 0, base: [] };
    cambiarFase("calibrando");
  }

  let mensaje = "";
  if (fase === "cargando") mensaje = "Prendiendo la cámara...";
  else if (fase === "calibrando") mensaje = "Pon cara tranquila 😐";
  else if (fase === "haciendo") mensaje = hayCara ? ejercicio.instruccion : "No te veo la carita, ponte frente a la cámara 🙈";

  return (
    <div className="path-screen">
      <Encabezado icono={ejercicio.icono} titulo={ejercicio.titulo} onBack={onBack} />
      <div className="repeat-stage ejercicio-stage">
        {fase === "error" ? (
          <div className="audio-only-error">{error}</div>
        ) : fase === "listo" ? (
          <>
            <div className="result-overlay-emoji" style={{ fontSize: 64 }}>🎉</div>
            <div className="repeat-hint">¡Muy bien! Completaste las {REPETICIONES} repeticiones.</div>
            <button className="path-cta" onClick={repetir}>Otra vez 🔁</button>
            <button className="path-cta" style={{ background: "transparent", color: "var(--text-secondary)", boxShadow: "none" }} onClick={onBack}>
              Elegir otro ejercicio
            </button>
          </>
        ) : (
          <>
            <div className="repeat-hint">{mensaje}</div>
            <div className="praxia-cam">
              <video ref={videoRef} className="praxia-video" muted playsInline />
              {fase === "haciendo" && (
                <div className="praxia-anillo" style={{ "--p": `${Math.round(progreso * 360)}deg` }}>
                  <span className="praxia-emoji">{ejercicio.icono}</span>
                </div>
              )}
            </div>
            {fase === "haciendo" && (
              <div className="nivel-barra" title="Qué tan cerca estás del gesto">
                <div className="nivel-relleno" style={{ width: `${Math.round(senal * 100)}%` }} />
              </div>
            )}
            {fase === "haciendo" && (
              <div className="praxia-reps" aria-label={`${reps} de ${REPETICIONES}`}>
                {Array.from({ length: REPETICIONES }, (_, i) => (
                  <span key={i} className={i < reps ? "rep-on" : "rep-off"}>{i < reps ? "⭐" : "○"}</span>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- SOPLO
function SoploStage({ ejercicio, onBack }) {
  const rafRef = useRef(null);
  const streamRef = useRef(null);
  const ctxRef = useRef(null);
  const faseRef = useRef("cargando");
  const calRef = useRef({ t0: 0, maximo: 0, umbral: 0.05 });
  const holdRef = useRef({ inicio: null, ultimoFuerte: 0 });

  const [fase, setFase] = useState("cargando"); // cargando | calibrando | soplando | listo | error
  const [error, setError] = useState(null);
  const [progreso, setProgreso] = useState(0);
  const [nivel, setNivel] = useState(0);

  const SOPLO_MS = 2500;

  function cambiarFase(f) {
    faseRef.current = f;
    setFase(f);
  }

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const ctx = new AudioCtx();
        ctxRef.current = ctx;
        const src = ctx.createMediaStreamSource(stream);
        const analizador = ctx.createAnalyser();
        analizador.fftSize = 1024;
        src.connect(analizador);
        const buf = new Float32Array(analizador.fftSize);

        calRef.current = { t0: performance.now(), maximo: 0, umbral: 0.05 };
        cambiarFase("calibrando");

        const tick = () => {
          rafRef.current = requestAnimationFrame(tick);
          analizador.getFloatTimeDomainData(buf);
          let suma = 0;
          for (let i = 0; i < buf.length; i++) suma += buf[i] * buf[i];
          const rms = Math.sqrt(suma / buf.length);
          setNivel(Math.min(rms * 6, 1));
          const ahora = performance.now();

          if (faseRef.current === "calibrando") {
            const c = calRef.current;
            c.maximo = Math.max(c.maximo, rms);
            if (ahora - c.t0 >= 1000) {
              c.umbral = Math.max(c.maximo * 3, 0.04);
              cambiarFase("soplando");
            }
            return;
          }
          if (faseRef.current !== "soplando") return;

          const h = holdRef.current;
          if (rms >= calRef.current.umbral) {
            h.ultimoFuerte = ahora;
            if (h.inicio === null) h.inicio = ahora;
          } else if (h.inicio !== null && ahora - h.ultimoFuerte > 400) {
            h.inicio = null; // dejó de soplar más de 0,4 s: empieza de nuevo
          }
          const prog = h.inicio === null ? 0 : Math.min((ahora - h.inicio) / SOPLO_MS, 1);
          setProgreso(prog);
          if (prog >= 1) {
            cambiarFase("listo");
            darPremio();
          }
        };
        rafRef.current = requestAnimationFrame(tick);
      } catch (e) {
        if (!cancelled) {
          setError(e?.name === "NotAllowedError"
            ? "Necesito permiso para usar el micrófono."
            : "No pude abrir el micrófono.");
          cambiarFase("error");
        }
      }
    })();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      ctxRef.current?.close().catch(() => {});
      ctxRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ejercicio.id]);

  function repetir() {
    holdRef.current = { inicio: null, ultimoFuerte: 0 };
    setProgreso(0);
    calRef.current = { ...calRef.current, t0: performance.now(), maximo: 0 };
    cambiarFase("calibrando");
  }

  return (
    <div className="path-screen">
      <Encabezado icono={ejercicio.icono} titulo={ejercicio.titulo} onBack={onBack} />
      <div className="repeat-stage ejercicio-stage">
        {fase === "error" ? (
          <div className="audio-only-error">{error}</div>
        ) : fase === "listo" ? (
          <>
            <div className="result-overlay-emoji" style={{ fontSize: 64 }}>🎉</div>
            <div className="repeat-hint">¡Apagaste la vela! 💨</div>
            <button className="path-cta" onClick={repetir}>Otra vez 🔁</button>
            <button className="path-cta" style={{ background: "transparent", color: "var(--text-secondary)", boxShadow: "none" }} onClick={onBack}>
              Elegir otro ejercicio
            </button>
          </>
        ) : (
          <>
            <div className="repeat-hint">
              {fase === "cargando" ? "Prendiendo el micrófono..."
                : fase === "calibrando" ? "Shhh, un segundito de silencio 🤫"
                  : ejercicio.instruccion}
            </div>
            <div className="vela">
              <div className="llama" style={{ transform: `scale(${1 - progreso * 0.9})`, opacity: 1 - progreso * 0.85 }}>🔥</div>
              <div className="vela-cuerpo" />
            </div>
            <div className="nivel-barra"><div className="nivel-relleno" style={{ width: `${Math.round(nivel * 100)}%` }} /></div>
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- VOZ
function VozStage({ ejercicio, onBack }) {
  const streamRef = useRef(null);
  const recorderRef = useRef(createAudioRecorder());
  const [micListo, setMicListo] = useState(false);
  const [micError, setMicError] = useState(null);
  const [grabando, setGrabando] = useState(false);
  const [procesando, setProcesando] = useState(false);
  const [resultado, setResultado] = useState(null);

  useEffect(() => {
    let cancelled = false;
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
  }, [ejercicio.id]);

  function empezar() {
    if (!streamRef.current) return;
    if (recorderRef.current.start(streamRef.current)) {
      setGrabando(true);
      setResultado(null);
    }
  }

  async function terminar() {
    setGrabando(false);
    const blob = recorderRef.current.stop();
    if (!blob) return;
    setProcesando(true);
    try {
      const r = await ejercicioVoz(getServerUrl(), blob, ejercicio.texto, ejercicio.modo);
      setResultado(r);
      if (r.valid && r.correcta) darPremio();
    } catch (e) {
      setResultado({ valid: false, reason: e.message || "Error al analizar el audio." });
    } finally {
      setProcesando(false);
    }
  }

  return (
    <div className="path-screen">
      <Encabezado icono={ejercicio.icono} titulo={ejercicio.titulo} onBack={onBack} />
      <div className="repeat-stage ejercicio-stage">
        <div className="repeat-hint">{ejercicio.instruccion}</div>
        <div className="repeat-word" style={{ fontSize: ejercicio.modo === "ritmo" ? 34 : 24, lineHeight: 1.3 }}>
          {ejercicio.mostrar}
        </div>

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
              {procesando ? "Analizando..." : grabando ? "Escuchando..." : "Toca para grabar"}
            </span>
          </div>
        )}

        {micListo && !procesando && (
          <button className="path-cta" onClick={grabando ? terminar : empezar}>
            {grabando ? "Listo, evaluar →" : "🎤 Grabar"}
          </button>
        )}

        {resultado && (
          resultado.valid ? (
            <div className="result-overlay-tip" style={{ marginTop: 8 }}>
              <div className="result-overlay-emoji">{resultado.correcta ? "😄" : "😅"}</div>
              <span className="result-overlay-tip-label">
                {resultado.correcta
                  ? (ejercicio.modo === "ritmo"
                    ? `¡Muy bien! Hiciste ${resultado.ciclos} vueltas de pa-ta-ka`
                    : "¡Lo lograste!")
                  : (ejercicio.modo === "ritmo"
                    ? `¡Casi! Hiciste ${resultado.ciclos} ${resultado.ciclos === 1 ? "vuelta" : "vueltas"} -- necesito 3`
                    : (resultado.palabras_flojas?.length
                      ? `¡Casi! Fíjate en «${resultado.palabras_flojas.join("», «")}» y dilo otra vez`
                      : "¡Casi! Dilo otra vez, despacito y claro"))}
              </span>
              {ejercicio.modo === "ritmo" && (
                <span className="result-overlay-tip-text">
                  Ritmo: {resultado.silabas_por_segundo} sílabas por segundo
                </span>
              )}
              {!resultado.correcta && (
                <button className="path-cta" style={{ marginTop: 10 }} onClick={empezar}>
                  🎤 Intentar de nuevo
                </button>
              )}
            </div>
          ) : (
            <div className="audio-only-error">{resultado.reason || "No se pudo analizar."}</div>
          )
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- PANTALLA
export default function EjerciciosScreen() {
  const [abierto, setAbierto] = useState(null);
  const ejercicio = EJERCICIOS.find((e) => e.id === abierto);

  if (ejercicio) {
    const volver = () => setAbierto(null);
    if (ejercicio.tipo === "praxia") return <PraxiaStage key={ejercicio.id} ejercicio={ejercicio} onBack={volver} />;
    if (ejercicio.tipo === "soplo") return <SoploStage key={ejercicio.id} ejercicio={ejercicio} onBack={volver} />;
    return <VozStage key={ejercicio.id} ejercicio={ejercicio} onBack={volver} />;
  }

  return (
    <div className="path-screen">
      <div className="path-header">
        <span className="path-title"><span className="path-mascot">🤸</span> Ejercicios</span>
      </div>
      <div className="racha-bar">
        <span className="racha-hint">Calienta la carita y la boca antes de practicar 💪</span>
      </div>
      <div className="etapa-list">
        {EJERCICIOS.map((e, i) => (
          <button
            key={e.id}
            className={`etapa-card etapa-color-${i % 4}`}
            onClick={() => setAbierto(e.id)}
            style={{ animationDelay: `${i * 0.06}s` }}
          >
            <span className="etapa-card-tag">{ETIQUETA_TIPO[e.tipo]}</span>
            <span className="etapa-card-word">
              <span className="etapa-card-icon">{e.icono}</span> {e.titulo}
            </span>
            <span className="etapa-card-action">Probar →</span>
          </button>
        ))}
      </div>
    </div>
  );
}
