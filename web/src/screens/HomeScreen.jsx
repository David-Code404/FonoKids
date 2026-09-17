import { useCallback, useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { checkHealth, getServerUrl, predict, setServerUrl as saveServerUrl, thumbnailUrl } from "../api.js";
import SettingsDialog from "../components/SettingsDialog.jsx";
import "./HomeScreen.css";

// Tope de duración de una toma -- coincide con MAX_FRAMES_BUFFER de
// probar_modelo.py (~150 frames a 25fps). Si el usuario se olvida de tocar
// para detener, esto corta solo en vez de mandar un clip cada vez más largo.
const MAX_RECORDING_MS = 6000;
const CONNECTION_REFRESH_MS = 20000;

// Overlay de puntos EN VIVO -- solo visual, no afecta la predicción real
// (que corre en el servidor con FAN). Mismo modelo y esquema de puntos
// que usaba PREVIEW_LIP_INDICES en home_screen.dart / probar_modelo.py
// (malla de 468 puntos de MediaPipe FaceMesh, 40 de ellos son labios).
const MEDIAPIPE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const MEDIAPIPE_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const PREVIEW_LIP_INDICES = new Set([
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78,
  95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
]);
const PREVIEW_EYE_CORNERS = new Set([33, 263]);
const LIVE_DETECT_EVERY = 2; // correr el detector del preview cada N frames

function pickMimeType() {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

/// Pestaña "Practicar" -- cámara en tiempo real + evaluación de
/// pronunciación por palabra.
export default function HomeScreen() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const autoStopTimerRef = useRef(null);
  const nextAttemptId = useRef(0);
  const pendingTimersRef = useRef([]);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const frameCountRef = useRef(0);
  const processingCountRef = useRef(0);
  // Cadena de promesas que garantiza que las predicciones se manden al
  // servidor de a una, nunca en paralelo (ver runPrediction más abajo).
  const predictQueueRef = useRef(Promise.resolve());

  const [cameraError, setCameraError] = useState(null);
  const [recording, setRecording] = useState(false);
  // Cantidad de predicciones corriendo en segundo plano -- a diferencia de un
  // booleano "predicting", esto permite que el usuario grabe la siguiente
  // toma sin esperar a que el servidor termine de analizar la anterior.
  const [processingCount, setProcessingCount] = useState(0);
  const [serverUrl, setServerUrlState] = useState(getServerUrl());
  const [connState, setConnState] = useState("unknown"); // unknown | ok | fail
  const [lastResult, setLastResult] = useState(null);
  const [lastError, setLastError] = useState(null);
  const [retryingPredict, setRetryingPredict] = useState(false);
  const [attempts, setAttempts] = useState([]);
  const [fadingOutIds, setFadingOutIds] = useState(new Set());
  const [unclearCount, setUnclearCount] = useState(0);
  const [showSettings, setShowSettings] = useState(false);

  const refreshConnection = useCallback(async (url) => {
    const ok = await checkHealth(url ?? serverUrl);
    setConnState(ok ? "ok" : "fail");
  }, [serverUrl]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setCameraError(null);
      } catch (e) {
        setCameraError(`No se pudo iniciar la cámara: ${e.message || e}`);
      }
    })();

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // Overlay de puntos EN VIVO -- carga el FaceLandmarker de MediaPipe una
  // vez y corre un loop de detección + dibujo sobre el <canvas>, sincronizado
  // con la resolución real del video (no con el tamaño CSS del preview, que
  // puede estar recortado por object-fit: cover).
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
        const landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MEDIAPIPE_MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numFaces: 1,
        });
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;
      } catch (e) {
        // Overlay puramente cosmético -- si falla (ej. sin internet para
        // bajar el modelo la primera vez), la app sigue funcionando igual,
        // solo sin los puntos dibujados.
        console.warn("No se pudo cargar el overlay de landmarks:", e);
      }
    })();

    function loop() {
      rafRef.current = requestAnimationFrame(loop);
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const landmarker = landmarkerRef.current;
      if (!video || !canvas || !landmarker || video.readyState < 2) return;

      frameCountRef.current += 1;
      if (frameCountRef.current % LIVE_DETECT_EVERY !== 0) return;

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const result = landmarker.detectForVideo(video, performance.now());
      const face = result.faceLandmarks?.[0];
      if (!face) return;

      for (let i = 0; i < face.length; i++) {
        const { x, y } = face[i];
        const px = x * canvas.width;
        const py = y * canvas.height;
        let radius = 1;
        let color = "rgba(255,0,0,0.5)";
        if (PREVIEW_LIP_INDICES.has(i)) {
          radius = 2.5;
          color = "#2ECC8F";
        } else if (PREVIEW_EYE_CORNERS.has(i)) {
          radius = 3.5;
          color = "#FFD84D";
        } else {
          continue; // el resto de la malla no aporta nada visualmente acá
        }
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }
    rafRef.current = requestAnimationFrame(loop);

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
    };
  }, []);

  useEffect(() => {
    refreshConnection(serverUrl);
    const id = setInterval(() => refreshConnection(serverUrl), CONNECTION_REFRESH_MS);
    return () => clearInterval(id);
  }, [serverUrl, refreshConnection]);

  useEffect(() => {
    return () => {
      pendingTimersRef.current.forEach(clearTimeout);
      if (autoStopTimerRef.current) clearTimeout(autoStopTimerRef.current);
    };
  }, []);

  // "correcta" viene DIRECTO del servidor (result.correcta: true/false/null)
  // -- ahí es donde se decide de verdad (softmax + margen, ver
  // server/main.py), no acá. null significa "no hubo confianza suficiente"
  // (MIN_CONFIDENCE_PROB/MARGIN) -- en ese caso no se registra como intento
  // real, solo se suma al contador de "no reconocidas", porque mostrar una
  // palabra "adivinada" con baja confianza confunde más de lo que ayuda.
  function registerAttempt(palabra, prob, captureFile, className, correcta) {
    if (correcta === null || correcta === undefined) {
      setUnclearCount((n) => n + 1);
      return;
    }
    const id = nextAttemptId.current++;
    // Foto real del momento (frame del clip guardado) -- si el servidor no
    // pudo guardarla, photoUrl queda null y el panel cae al ícono genérico.
    const photoUrl = captureFile ? thumbnailUrl(serverUrl, className, captureFile) : null;
    const entry = { id, palabra, prob, correcta, time: new Date(), photoUrl };
    setAttempts((prev) => [entry, ...prev]);
  }

  function dismissAttempt(id) {
    setAttempts((prev) => prev.filter((c) => c.id !== id));
    setFadingOutIds((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }

  function clearAttempts() {
    setAttempts([]);
    setFadingOutIds(new Set());
  }

  // Corre la predicción de un clip ya grabado SIN bloquear la UI -- se llama
  // "fire and forget" desde stopAndPredict, así el botón de grabar queda
  // libre de nuevo apenas se corta la toma, en vez de esperar los ~1-90s que
  // puede tardar el servidor (HRNet + Conformer) en responder.
  //
  // IMPORTANTE: nunca se manda una predicción al servidor mientras otra
  // sigue en vuelo -- se ENCADENAN en predictQueueRef, una atrás de la otra.
  // La GPU (6GB) solo puede procesar una a la vez igual (ver el semáforo en
  // server/main.py), así que mandar varias en paralelo desde acá no gana
  // nada: solo hace que la 2da/3ra empiecen a contar su propio timeout de
  // 60s mientras siguen esperando en cola del lado del servidor, y terminen
  // fallando por timeout aunque el servidor las esté por responder bien.
  // Encadenando client-side, cada request arranca su cronómetro recién
  // cuando el servidor la empieza a procesar de verdad.
  const runPrediction = useCallback(
    (blob, filename) => {
      processingCountRef.current += 1;
      setProcessingCount(processingCountRef.current);

      const run = async () => {
        try {
          const result = await predict(serverUrl, blob, filename, () => setRetryingPredict(true));
          setRetryingPredict(false);
          setLastResult(result);
          setLastError(null);
          setConnState("ok");
          if (result.valid && result.palabra) {
            registerAttempt(result.palabra, result.prob ?? 0, result.capture_file, result.class_name, result.correcta);
          }
        } catch (e) {
          setRetryingPredict(false);
          setLastError(e.message || "Error inesperado hablando con el servidor.");
          setConnState("fail");
          // Si el servidor se cuelga se reinicia solo en ~20-30s (ver
          // server/run.sh) -- sin esto, el chip "Sin conexión" se queda así
          // hasta el próximo chequeo automático (cada CONNECTION_REFRESH_MS,
          // hasta 20s más de espera). Reintentar un poco antes acorta esa
          // ventana en la que la app parece caída aunque ya haya vuelto.
          setTimeout(() => refreshConnection(), 15000);
        } finally {
          processingCountRef.current -= 1;
          setProcessingCount(processingCountRef.current);
        }
      };

      const jobPromise = predictQueueRef.current.then(run);
      predictQueueRef.current = jobPromise;
      return jobPromise;
    },
    [serverUrl]
  );

  const stopAndPredict = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;

    if (autoStopTimerRef.current) {
      clearTimeout(autoStopTimerRef.current);
      autoStopTimerRef.current = null;
    }

    const stopped = new Promise((resolve) => {
      recorder.onstop = resolve;
    });
    recorder.stop();
    await stopped;

    // Cortar la grabación es instantáneo -- apenas termina, el botón queda
    // libre para la siguiente toma. La predicción sigue su curso sola.
    setRecording(false);

    const mimeType = recorder.mimeType || "video/webm";
    const blob = new Blob(chunksRef.current, { type: mimeType });
    const ext = mimeType.includes("mp4") ? "mp4" : "webm";
    runPrediction(blob, `clip.${ext}`);
  }, [runPrediction]);

  const startRecording = useCallback(() => {
    const stream = streamRef.current;
    if (!stream) return;
    setLastResult(null);
    setLastError(null);

    try {
      const mimeType = pickMimeType();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);

      autoStopTimerRef.current = setTimeout(() => {
        stopAndPredict();
      }, MAX_RECORDING_MS);
    } catch (e) {
      setLastError(`No se pudo empezar a grabar: ${e.message || e}`);
    }
  }, [stopAndPredict]);

  function toggleRecording() {
    // Ya NO se bloquea por processingCount -- se puede grabar la siguiente
    // toma aunque la anterior todavía se esté analizando en el servidor.
    if (!recording) {
      startRecording();
    } else {
      stopAndPredict();
    }
  }

  function handleSaveSettings(url) {
    const trimmed = url.trim();
    if (trimmed) {
      saveServerUrl(trimmed);
      setServerUrlState(trimmed);
      setConnState("unknown");
      refreshConnection(trimmed);
    }
    setShowSettings(false);
  }

  const connLabel = { ok: "Servidor OK", fail: "Sin conexión", unknown: "Conectando..." }[connState];
  const connClass = { ok: "conn-ok", fail: "conn-fail", unknown: "conn-unknown" }[connState];

  return (
    <div className="practice-screen">
      <div className="practice-main">
        <div className="practice-header">
          <div className="brand">
            <span className="brand-mascot">🗣️</span>
            <span>SpeakShadow</span>
          </div>
          <div className="header-actions">
            <button className={`conn-chip ${connClass}`} onClick={() => refreshConnection()}>
              <span className="conn-dot" />
              {connLabel}
            </button>
            <button className="icon-button" onClick={() => setShowSettings(true)} title="Configuración servidor">
              ⚙
            </button>
          </div>
        </div>

        <div className="practice-stage">
          <div className={`camera-frame ${recording ? "is-recording" : ""}`}>
            {cameraError ? (
              <div className="center-message">{cameraError}</div>
            ) : (
              <>
                <video ref={videoRef} className="camera-preview" autoPlay playsInline muted />
                <canvas ref={canvasRef} className="camera-overlay" />
              </>
            )}
          </div>

          <div className="stage-hint">
            {retryingPredict
              ? "⏳ El servidor se colgó, esperando a que vuelva para reintentar solo..."
              : lastError
              ? `⚠ ${lastError}`
              : recording
              ? "Grabando... tocá para terminar"
              : processingCount > 0
              ? `Analizando ${processingCount} intento(s)...`
              : "Decí la palabra y tocá el botón para grabar"}
          </div>

          <div className="record-button-wrap">
            <button
              className={`record-button ${recording ? "recording" : ""}`}
              onClick={toggleRecording}
              disabled={!!cameraError}
            >
              <span className={recording ? "square" : "circle"} />
            </button>
            {processingCount > 0 && <span className="processing-badge">{processingCount}</span>}
          </div>
        </div>
      </div>

      <PracticeHistoryPanel
        attempts={attempts}
        fadingOutIds={fadingOutIds}
        unclearCount={unclearCount}
        onDismiss={dismissAttempt}
        onClear={attempts.length ? clearAttempts : null}
      />

      {!retryingPredict && lastResult && (
        <ResultOverlay result={lastResult} onClose={() => setLastResult(null)} />
      )}

      {showSettings && (
        <SettingsDialog
          currentUrl={serverUrl}
          onSave={handleSaveSettings}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}

/// Tarjeta grande centrada con el resultado del intento -- se tapa sola
/// apenas arranca la próxima grabación (ver startRecording), o el chico
/// puede tocar afuera para cerrarla antes.
function ResultOverlay({ result, onClose }) {
  if (!result.valid) {
    return (
      <div className="result-overlay" onClick={onClose}>
        <div className="result-overlay-card" onClick={(e) => e.stopPropagation()}>
          <div className="result-overlay-emoji">🤔</div>
          <div className="result-overlay-message">{result.reason || "Toma no válida, repetí."}</div>
          <button className="result-overlay-button neutral" onClick={onClose}>
            Entendido
          </button>
        </div>
      </div>
    );
  }

  const palabra = (result.palabra || "?").replaceAll("_", " ");

  // La decisión (bien/mal dicha) es del servidor (softmax + margen sobre
  // clases "<palabra>_correcto"/"<palabra>_incorrecto", ver server/main.py)
  // -- no se recalcula acá.
  if (!result.confianza_suficiente) {
    return (
      <div className="result-overlay" onClick={onClose}>
        <div className="result-overlay-card" onClick={(e) => e.stopPropagation()}>
          <div className="result-overlay-emoji">👂</div>
          <div className="result-overlay-message">No pude escucharla bien -- ¡repetí la toma!</div>
          <button className="result-overlay-button neutral" onClick={onClose}>
            Dale, de nuevo
          </button>
        </div>
      </div>
    );
  }

  const pct = ((result.prob ?? 0) * 100).toFixed(0);
  return (
    <div className="result-overlay" onClick={onClose}>
      <div className={`result-overlay-card ${result.correcta ? "success" : "retry"}`} onClick={(e) => e.stopPropagation()}>
        <div className="result-overlay-emoji">{result.correcta ? "🎉" : "🔁"}</div>
        <div className="result-overlay-word">{palabra}</div>
        {result.correcta ? (
          <div className="result-overlay-message">¡Muy bien dicho! -- {pct}% de seguridad</div>
        ) : (
          <div className="result-overlay-message">Casi... practiquemos de nuevo -- {pct}%</div>
        )}
        <button className={`result-overlay-button ${result.correcta ? "success" : "retry"}`} onClick={onClose}>
          {result.correcta ? "¡Genial! 🌟" : "Intentar de nuevo"}
        </button>
      </div>
    </div>
  );
}

/// Foto real del intento (frame del clip guardado en el servidor) -- cae
/// al ícono genérico si no hay foto o si falla la carga.
function AttemptAvatar({ photoUrl }) {
  const [failed, setFailed] = useState(false);
  if (!photoUrl || failed) {
    return <div className="capture-avatar">🗣</div>;
  }
  return (
    <div className="capture-avatar capture-avatar-photo">
      <img src={photoUrl} alt="Foto del intento" onError={() => setFailed(true)} />
    </div>
  );
}

function PracticeHistoryPanel({ attempts, fadingOutIds, unclearCount, onDismiss, onClear }) {
  const correctCount = attempts.filter((c) => c.correcta).length;
  const incorrectCount = attempts.filter((c) => !c.correcta).length;
  return (
    <div className="captures-panel">
      <div className="captures-header">
        <div className="captures-title">
          <div className="captures-title-text">⭐ Racha de hoy</div>
          <div className="captures-subtitle">{attempts.length} intentos en esta sesión</div>
        </div>
        {onClear && (
          <button className="icon-button" onClick={onClear} title="Limpiar sesión">
            🗑
          </button>
        )}
      </div>
      <div className="captures-note">
        Cada intento se evalúa por separado -- practicá las veces que quieras.
      </div>
      <div className="mini-stats">
        <div className="mini-stat correct">✅ {correctCount} Bien dichas</div>
        <div className="mini-stat risk">🔁 {incorrectCount} Para practicar</div>
        <div className="mini-stat muted">❔ {unclearCount} No reconocidas</div>
      </div>
      <div className="captures-list">
        {attempts.length === 0 ? (
          <div className="captures-empty">
            Todavía no practicaste ninguna palabra.
            <br />
            Grabate diciendo una y va a aparecer acá.
          </div>
        ) : (
          attempts.map((entry) => (
            <div
              key={entry.id}
              className={`capture-card ${entry.correcta ? "" : "risk"} ${fadingOutIds.has(entry.id) ? "fading" : ""}`}
            >
              <AttemptAvatar photoUrl={entry.photoUrl} />
              <div className="capture-info">
                <div className="capture-row">
                  <span className="capture-word">{entry.palabra.replaceAll("_", " ")}</span>
                  <button className="dismiss-button" onClick={() => onDismiss(entry.id)}>
                    ✕
                  </button>
                </div>
                <div className="capture-sub">
                  {entry.time.getHours().toString().padStart(2, "0")}:
                  {entry.time.getMinutes().toString().padStart(2, "0")} ·{" "}
                  {(entry.prob * 100).toFixed(0)}%
                </div>
                <div className={`capture-tag ${entry.correcta ? "" : "risk"}`}>
                  {entry.correcta ? "✅ Bien dicha" : "🔁 Para practicar de nuevo"}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
