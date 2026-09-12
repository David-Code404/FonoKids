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

// Modo "vigilancia" (cámara de seguridad) -- un solo tap arranca, y la
// cámara graba segmentos cortos encadenados uno tras otro sola, mandando
// cada uno a predecir en segundo plano, hasta que se apaga con otro tap.
const VIGILANCE_SEGMENT_MS = 5000; // duración de cada segmento
// La GPU procesa UNA predicción a la vez (ver semáforo en server/main.py) y
// puede tardar bastante más que un segmento -- si dejáramos encolar todos
// los segmentos sin límite, la cola crecería sin parar y las alertas
// llegarían cada vez más tarde respecto al momento real. Por eso, pasado
// este tope de predicciones pendientes, se DESCARTAN segmentos nuevos en
// vez de acumularlos -- es vigilancia en vivo, no un archivo para revisar
// después. En 1: mientras una predicción sigue en curso (adentro del
// modelo), cualquier segmento nuevo se descarta directo -- nunca hay una
// segunda esperando en fila para el modelo, solo se vuelve a grabar/mandar
// recién cuando la anterior termina de verdad.
const MAX_QUEUED_PREDICTIONS = 1;

// Pausa entre que termina una predicción y arranca a grabar el siguiente
// segmento -- le da un respiro a la GPU en vez de mandarle pedidos pegados
// uno atrás de otro sin parar (ver nota larga en finishVigilanceSegment).
const VIGILANCE_COOLDOWN_MS = 2000;

// Overlay de puntos EN VIVO -- solo visual, no afecta la predicción real
// (que corre en el servidor con HRNet). Mismo modelo y esquema de puntos
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

/// Pestaña "Grabar" -- cámara en tiempo real + predicción, igual que
/// home_screen.dart (HomeScreen).
export default function HomeScreen() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const autoStopTimerRef = useRef(null);
  const vigilanceCooldownTimerRef = useRef(null);
  const nextCaptureId = useRef(0);
  const pendingTimersRef = useRef([]);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const frameCountRef = useRef(0);
  // Espejos en ref de processingCount/vigilanceMode -- se leen de forma
  // síncrona dentro del loop de segmentos (finishVigilanceSegment), donde
  // un closure sobre el state de React podría quedar desactualizado.
  const processingCountRef = useRef(0);
  const vigilanceModeRef = useRef(false);
  // Cadena de promesas que garantiza que las predicciones se manden al
  // servidor de a una, nunca en paralelo (ver runPrediction más abajo).
  const predictQueueRef = useRef(Promise.resolve());

  const [cameraError, setCameraError] = useState(null);
  const [recording, setRecording] = useState(false);
  const [vigilanceMode, setVigilanceMode] = useState(false);
  // Cantidad de predicciones corriendo en segundo plano -- a diferencia de un
  // booleano "predicting", esto permite que el usuario grabe la siguiente
  // toma sin esperar a que el servidor termine de analizar la anterior (así
  // no se traba la app mientras varias personas hablan seguido).
  const [processingCount, setProcessingCount] = useState(0);
  const [serverUrl, setServerUrlState] = useState(getServerUrl());
  const [connState, setConnState] = useState("unknown"); // unknown | ok | fail
  const [lastResult, setLastResult] = useState(null);
  const [lastError, setLastError] = useState(null);
  const [retryingPredict, setRetryingPredict] = useState(false);
  const [captures, setCaptures] = useState([]);
  const [fadingOutIds, setFadingOutIds] = useState(new Set());
  const [discardedCount, setDiscardedCount] = useState(0);
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
          color = "#35C9C1";
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
      if (vigilanceCooldownTimerRef.current) clearTimeout(vigilanceCooldownTimerRef.current);
    };
  }, []);

  // isRiesgo viene DIRECTO del servidor (result.es_frase_de_riesgo) -- ahí
  // es donde se decide de verdad (softmax + margen + distancia al centroide
  // de la clase, ver server/main.py), no acá. Si no es riesgo, ni siquiera
  // mostramos la palabra que el modelo creyó reconocer -- el modelo SIEMPRE
  // tiene que elegir alguna de las 20 clases entrenadas aunque no sea
  // ninguna de verdad, así que mostrar esa palabra "adivinada" para algo
  // que en realidad no dijiste confunde más de lo que ayuda. Se descarta
  // directo, sin tarjeta ni fade.
  function registerCapture(word, prob, captureFile, isRiesgo) {
    if (!isRiesgo) {
      setDiscardedCount((n) => n + 1);
      return;
    }
    const id = nextCaptureId.current++;
    // Foto real del momento (frame del clip guardado) -- si el servidor no
    // pudo guardarla, photoUrl queda null y el panel cae al ícono genérico.
    const photoUrl = captureFile ? thumbnailUrl(serverUrl, word, captureFile) : null;
    const entry = { id, word, prob, isRisk: true, time: new Date(), photoUrl };
    setCaptures((prev) => [entry, ...prev]);
  }

  function dismissCapture(id) {
    setCaptures((prev) => prev.filter((c) => c.id !== id));
    setFadingOutIds((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }

  function clearCaptures() {
    setCaptures([]);
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
          if (result.valid && result.word) {
            registerCapture(result.word, result.prob ?? 0, result.capture_file, !!result.es_frase_de_riesgo);
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

      // Devuelve la promesa para que el modo vigilancia pueda esperar a que
      // ESTA predicción puntual termine antes de grabar el siguiente
      // segmento (ver finishVigilanceSegment) -- la grabación manual la
      // ignora, sigue siendo "fire and forget" como antes.
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

  // --- Modo vigilancia: segmentos encadenados sin volver a tocar nada ---

  // Ref hacia la última versión de finishVigilanceSegment -- así
  // startVigilanceSegment puede quedar memoizado para siempre (deps [], no
  // depende de nada reactivo) sin arrastrar un closure viejo si serverUrl
  // cambia (ej. el usuario edita la URL del servidor en Configuración
  // mientras la vigilancia está prendida).
  const finishVigilanceSegmentRef = useRef(() => {});

  const startVigilanceSegment = useCallback(() => {
    const stream = streamRef.current;
    if (!stream || !vigilanceModeRef.current) return;

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

      autoStopTimerRef.current = setTimeout(
        () => finishVigilanceSegmentRef.current(),
        VIGILANCE_SEGMENT_MS
      );
    } catch (e) {
      setLastError(`No se pudo grabar el segmento: ${e.message || e}`);
      vigilanceModeRef.current = false;
      setVigilanceMode(false);
    }
  }, []);

  const finishVigilanceSegment = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;

    const stopped = new Promise((resolve) => {
      recorder.onstop = resolve;
    });
    recorder.stop();
    await stopped;
    setRecording(false);

    const mimeType = recorder.mimeType || "video/webm";
    const blob = new Blob(chunksRef.current, { type: mimeType });
    const ext = mimeType.includes("mp4") ? "mp4" : "webm";

    // Si el servidor ya tiene demasiadas predicciones pendientes, este
    // segmento se descarta en vez de sumarse a una cola que crecería sin
    // parar -- vigilancia en vivo, no un archivo para revisar después.
    if (processingCountRef.current < MAX_QUEUED_PREDICTIONS) {
      const jobPromise = runPrediction(blob, `segment.${ext}`);
      // OJO: acá NO se encadena el siguiente segmento de una -- se espera a
      // que ESTA predicción termine + una pausa corta (VIGILANCE_COOLDOWN_MS)
      // antes de volver a grabar. Mandar predicciones pegadas una atrás de
      // otra sin ninguna pausa es justo lo que parece disparar el cuelgue de
      // driver visto en uso intenso -- con probar_modelo.py nunca pasa
      // porque ahí SIEMPRE hay una pausa natural (grabás vos a mano). Esto
      // sacrifica cobertura continua (hay un hueco corto sin grabar) a
      // cambio de no freír la GPU.
      jobPromise.finally(() => {
        if (vigilanceModeRef.current) {
          vigilanceCooldownTimerRef.current = setTimeout(() => {
            vigilanceCooldownTimerRef.current = null;
            startVigilanceSegment();
          }, VIGILANCE_COOLDOWN_MS);
        }
      });
      return;
    }

    // Se descartó el segmento (cola llena) -- no hay ninguna predicción en
    // curso que esperar, así que se puede volver a grabar ya mismo.
    if (vigilanceModeRef.current) {
      startVigilanceSegment();
    }
  }, [runPrediction, startVigilanceSegment]);

  useEffect(() => {
    finishVigilanceSegmentRef.current = finishVigilanceSegment;
  }, [finishVigilanceSegment]);

  function toggleVigilance() {
    if (vigilanceMode) {
      vigilanceModeRef.current = false;
      setVigilanceMode(false);
      if (autoStopTimerRef.current) {
        clearTimeout(autoStopTimerRef.current);
        autoStopTimerRef.current = null;
      }
      // Este es el timer que faltaba cancelar -- la pausa entre que termina
      // una predicción y arranca a grabar el siguiente segmento (ver
      // finishVigilanceSegment). Sin esto, apagar vigilancia justo durante
      // esa pausa no cortaba nada de verdad: 2s después igual arrancaba a
      // grabar un segmento nuevo solo.
      if (vigilanceCooldownTimerRef.current) {
        clearTimeout(vigilanceCooldownTimerRef.current);
        vigilanceCooldownTimerRef.current = null;
      }
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        // Se apaga sin mandar el segmento a medio grabar -- descarta y listo.
        recorder.onstop = null;
        recorder.stop();
      }
      setRecording(false);
    } else {
      setLastResult(null);
      setLastError(null);
      vigilanceModeRef.current = true;
      setVigilanceMode(true);
      startVigilanceSegment();
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
    <div className="home-screen">
      <div className="camera-area">
        {cameraError ? (
          <div className="center-message">{cameraError}</div>
        ) : (
          <>
            <video ref={videoRef} className="camera-preview" autoPlay playsInline muted />
            <canvas ref={canvasRef} className="camera-overlay" />
          </>
        )}

        <div className="top-bar">
          <button className={`conn-chip ${connClass}`} onClick={() => refreshConnection()}>
            <span className="conn-dot" />
            {connLabel}
          </button>
          <div className="top-bar-actions">
            <button
              className={`vigilance-toggle ${vigilanceMode ? "active" : ""}`}
              onClick={toggleVigilance}
              disabled={recording && !vigilanceMode}
              title="Modo vigilancia: graba y predice sola en segmentos, sin volver a tocar nada"
            >
              👁 {vigilanceMode ? "Vigilando" : "Vigilancia"}
            </button>
            <button className="icon-button" onClick={() => setShowSettings(true)} title="Configuración servidor">
              ⚙
            </button>
          </div>
        </div>

        <div className="status-area">
          {retryingPredict && (
            <div className="card">⏳ El servidor se colgó, esperando a que vuelva para reintentar solo...</div>
          )}
          {!retryingPredict && lastError && <div className="card error-card">⚠ {lastError}</div>}
          {!retryingPredict && lastResult && <ResultCard result={lastResult} />}
        </div>

        <div className="record-control">
          <div className="record-hint">
            {vigilanceMode
              ? `👁 Vigilando -- segmento cada ${VIGILANCE_SEGMENT_MS / 1000}s (tocá para apagar)`
              : recording
              ? "Grabando... tocá para terminar"
              : processingCount > 0
              ? `Tocá para grabar (analizando ${processingCount} en 2do plano...)`
              : "Tocá para grabar"}
          </div>
          <div className="record-button-wrap">
            <button
              className={`record-button ${recording || vigilanceMode ? "recording" : ""}`}
              onClick={vigilanceMode ? toggleVigilance : toggleRecording}
              disabled={!!cameraError}
            >
              <span className={recording || vigilanceMode ? "square" : "circle"} />
            </button>
            {processingCount > 0 && <span className="processing-badge">{processingCount}</span>}
          </div>
        </div>
      </div>

      <CapturesPanel
        captures={captures}
        fadingOutIds={fadingOutIds}
        discardedCount={discardedCount}
        onDismiss={dismissCapture}
        onClear={captures.length ? clearCaptures : null}
      />

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

function ResultCard({ result }) {
  if (!result.valid) {
    return <div className="card error-card">⚠ {result.reason || "Toma no válida, repetí."}</div>;
  }
  // La decisión de riesgo es del servidor (softmax + margen + distancia al
  // centroide de la clase) -- no se recalcula acá. Si no es riesgo, no se
  // muestra la palabra "adivinada": el modelo siempre elige una de las 20
  // clases aunque no hayas dicho ninguna, así que mostrarla como si fuera
  // un resultado real confunde más de lo que ayuda.
  if (!result.es_frase_de_riesgo) {
    return (
      <div className="card result-card">
        <div className="result-badge no-risk">Palabra desconocida -- descartada</div>
      </div>
    );
  }
  const pct = ((result.prob ?? 0) * 100).toFixed(1);
  return (
    <div className="card result-card">
      <div className="result-word">{(result.word || "?").replaceAll("_", " ")}</div>
      <div className="result-badge risk">Frase de riesgo -- {pct}%</div>
    </div>
  );
}

/// Foto real de la captura (frame del clip guardado en el servidor) -- cae
/// al ícono genérico si no hay foto (frase no guardada) o si falla la carga.
function CaptureAvatar({ photoUrl }) {
  const [failed, setFailed] = useState(false);
  if (!photoUrl || failed) {
    return <div className="capture-avatar">👤</div>;
  }
  return (
    <div className="capture-avatar capture-avatar-photo">
      <img src={photoUrl} alt="Foto de la persona" onError={() => setFailed(true)} />
    </div>
  );
}

function CapturesPanel({ captures, fadingOutIds, discardedCount, onDismiss, onClear }) {
  const riskCount = captures.filter((c) => c.isRisk).length;
  return (
    <div className="captures-panel">
      <div className="captures-header">
        <div className="captures-title">
          <div className="captures-title-text">Personas capturadas</div>
          <div className="captures-subtitle">{captures.length} en esta sesión</div>
        </div>
        {onClear && (
          <button className="icon-button" onClick={onClear} title="Limpiar sesión">
            🗑
          </button>
        )}
      </div>
      <div className="captures-note">
        Cada detección es independiente: todavía no reconoce si dos capturas son la misma persona.
      </div>
      <div className="mini-stats">
        <div className="mini-stat risk">⚠ {riskCount} De riesgo</div>
        <div className="mini-stat muted">⏱ {discardedCount} Descartadas</div>
      </div>
      <div className="captures-list">
        {captures.length === 0 ? (
          <div className="captures-empty">
            Todavía no se detectó a nadie.
            <br />
            Cuando alguien hable, va a aparecer acá.
          </div>
        ) : (
          captures.map((entry) => (
            <div
              key={entry.id}
              className={`capture-card ${entry.isRisk ? "risk" : ""} ${fadingOutIds.has(entry.id) ? "fading" : ""}`}
            >
              <CaptureAvatar photoUrl={entry.photoUrl} />
              <div className="capture-info">
                <div className="capture-row">
                  <span className="capture-word">{entry.word.replaceAll("_", " ")}</span>
                  <button className="dismiss-button" onClick={() => onDismiss(entry.id)}>
                    ✕
                  </button>
                </div>
                <div className="capture-sub">
                  {entry.photoUrl ? "Foto real de este momento" : "Persona no identificada"}
                </div>
                <div className="capture-sub">
                  {entry.time.getHours().toString().padStart(2, "0")}:
                  {entry.time.getMinutes().toString().padStart(2, "0")} ·{" "}
                  {(entry.prob * 100).toFixed(0)}%
                </div>
                <div className={`capture-tag ${entry.isRisk ? "risk" : ""}`}>
                  {entry.isRisk ? "Guardado en Capturas" : "Descartando..."}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
