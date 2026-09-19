import { useCallback, useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import {
  checkHealth, getHealth, getServerUrl, predict,
  setServerUrl as saveServerUrl, thumbnailUrl, wordsFromClasses,
} from "../api.js";
import { iconFor } from "../wordIcons.js";
import SettingsDialog from "../components/SettingsDialog.jsx";
import "./HomeScreen.css";

// Tope de duración de una toma -- coincide con MAX_FRAMES_BUFFER de
// probar_modelo.py (~150 frames a 25fps). Si el usuario se olvida de tocar
// para detener, esto corta solo en vez de mandar un clip cada vez más largo.
const MAX_RECORDING_MS = 6000;
const CONNECTION_REFRESH_MS = 20000;

// Modo "en vivo" -- a diferencia de la vieja Vigilancia (sacada del
// proyecto: grababa en segundo plano sin que nadie la prendiera a
// propósito, un problema de privacidad para este uso), este modo lo prende
// el propio chico/persona que está practicando, mientras sigue mirando la
// cámara -- graba en segmentos cortos encadenados y da feedback rápido de
// cada uno, sin tener que tocar "grabar" después de cada intento.
const LIVE_SEGMENT_MS = 3000;
const LIVE_COOLDOWN_MS = 1200;
// En modo en vivo el resultado se cierra solo (no bloquea esperando un tap)
// para no cortar el flujo de práctica continua.
const LIVE_RESULT_AUTOCLOSE_MS = 2500;

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

// Arma un .wav de 16 bits mono a partir de los chunks Float32 que entrega
// el ScriptProcessorNode -- header RIFF/WAVE de 44 bytes escrito a mano
// (sin librerías: es un formato simple y así queda controlado 100%).
function encodeWav(chunks, sampleRate) {
  let totalLength = 0;
  for (const c of chunks) totalLength += c.length;

  const pcm16 = new Int16Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++) {
      const s = Math.max(-1, Math.min(1, chunk[i]));
      pcm16[offset++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
  }

  const buffer = new ArrayBuffer(44 + pcm16.length * 2);
  const view = new DataView(buffer);
  const writeStr = (pos, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(pos + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + pcm16.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (16 bits * 1 canal)
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits por muestra
  writeStr(36, "data");
  view.setUint32(40, pcm16.length * 2, true);
  new Int16Array(buffer, 44).set(pcm16);

  return new Blob([buffer], { type: "audio/wav" });
}

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
export default function HomeScreen({ initialWord = null, onBack = null }) {
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
  // Refs espejo de liveMode/targetWord -- se leen desde dentro del loop de
  // segmentos (finishLiveSegment), donde un closure sobre el state de React
  // podría quedar desactualizado entre renders.
  const liveModeRef = useRef(false);
  const liveCooldownTimerRef = useRef(null);
  const finishLiveSegmentRef = useRef(() => {});
  const targetWordRef = useRef(null);
  // Captura de audio en paralelo al video -- ver startAudioCapture/
  // stopAudioCapture más abajo. Un ScriptProcessorNode en vez de
  // MediaRecorder para el audio: MediaRecorder graba webm/opus, que el
  // servidor NO puede decodificar (torchaudio ahí solo tiene el backend
  // "soundfile", sin soporte de opus) -- grabando PCM crudo y armando el
  // .wav a mano (mismo criterio que grabar_video_continuo.py, que usa
  // sounddevice + wave por la misma razón) se evita ese problema de raíz.
  const audioCtxRef = useRef(null);
  const audioProcessorRef = useRef(null);
  const audioSourceRef = useRef(null);
  const audioChunksRef = useRef([]);

  const [cameraError, setCameraError] = useState(null);
  const [recording, setRecording] = useState(false);
  const [liveMode, setLiveMode] = useState(false);
  const [targetWords, setTargetWords] = useState([]);
  const [targetWord, setTargetWord] = useState(null);
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
          audio: true,
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

  // Palabras objetivo -- se sacan de las clases que el modelo cargado
  // realmente reconoce (GET /health), no de una lista hardcodeada acá, así
  // si el checkpoint cambia (nuevas palabras entrenadas) esto se actualiza
  // solo sin tocar código.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const health = await getHealth(serverUrl);
      if (cancelled || !health) return;
      const words = wordsFromClasses(health.classes);
      setTargetWords(words);
      // Si llegamos acá desde el camino de etapas (initialWord), esa
      // palabra manda -- no la pisa la primera de la lista del server.
      setTargetWord((prev) =>
        initialWord && words.includes(initialWord) ? initialWord
        : prev && words.includes(prev) ? prev
        : words[0] ?? null
      );
    })();
    return () => {
      cancelled = true;
    };
  }, [serverUrl, connState]);

  useEffect(() => {
    targetWordRef.current = targetWord;
  }, [targetWord]);

  useEffect(() => {
    return () => {
      pendingTimersRef.current.forEach(clearTimeout);
      if (autoStopTimerRef.current) clearTimeout(autoStopTimerRef.current);
      if (liveCooldownTimerRef.current) clearTimeout(liveCooldownTimerRef.current);
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
    (blob, filename, audioBlob) => {
      processingCountRef.current += 1;
      setProcessingCount(processingCountRef.current);

      const run = async () => {
        try {
          const result = await predict(
            serverUrl, blob, filename, () => setRetryingPredict(true),
            audioBlob, targetWordRef.current
          );
          setRetryingPredict(false);
          // Veredicto ÚNICO combinado (labios + sonido) que ya calculó el
          // servidor -- lo mapeamos sobre "correcta" para que el resto de
          // la UI (overlay de resultado, tarjetas del historial, contador
          // de racha) siga funcionando igual, sin tener que enterarse de
          // que ahora hay dos señales detrás en vez de una.
          if (result.valid && result.veredicto_final) {
            result.correcta =
              result.veredicto_final === "bien" ? true
              : result.veredicto_final === "a_practicar" ? false
              : null;
          }
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

  // Arranca la captura de audio crudo (PCM) EN PARALELO al MediaRecorder de
  // video -- mismo stream, pista de audio aparte. Si el micrófono no está
  // disponible (permiso denegado, sin hardware), no rompe nada: sigue
  // grabando solo video, como antes de agregar esto.
  function startAudioCapture() {
    audioChunksRef.current = [];
    const stream = streamRef.current;
    const audioTrack = stream?.getAudioTracks?.()[0];
    if (!audioTrack) return;

    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      const source = ctx.createMediaStreamSource(new MediaStream([audioTrack]));
      // ScriptProcessorNode está deprecado pero sigue soportado en todos
      // los navegadores evergreen -- AudioWorklet exige cargar un archivo
      // aparte, innecesario para algo tan simple como juntar PCM crudo.
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => {
        audioChunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      // Conectar a destination es necesario para que Chrome/Firefox sigan
      // llamando onaudioprocess -- el audio real no se re-emite al usuario
      // porque la pista que llega acá nunca pasa por un <audio>/<video> con
      // sonido activado (el <video> de preview está muted).
      processor.connect(ctx.destination);
      audioCtxRef.current = ctx;
      audioSourceRef.current = source;
      audioProcessorRef.current = processor;
    } catch (e) {
      console.warn("No se pudo iniciar la captura de audio:", e);
    }
  }

  // Corta la captura y devuelve un Blob .wav ya armado, o null si no había
  // audio (mismo criterio "degrada con gracia" que el resto de la app).
  function stopAudioCapture() {
    const ctx = audioCtxRef.current;
    const processor = audioProcessorRef.current;
    const source = audioSourceRef.current;
    const chunks = audioChunksRef.current;
    audioCtxRef.current = null;
    audioProcessorRef.current = null;
    audioSourceRef.current = null;
    audioChunksRef.current = [];

    if (!ctx) return null;
    const sampleRate = ctx.sampleRate;
    try {
      processor?.disconnect();
      source?.disconnect();
      ctx.close();
    } catch {
      // no crítico -- el contexto se cierra solo eventualmente igual
    }
    if (chunks.length === 0) return null;
    return encodeWav(chunks, sampleRate);
  }

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
    const audioBlob = stopAudioCapture();

    // Cortar la grabación es instantáneo -- apenas termina, el botón queda
    // libre para la siguiente toma. La predicción sigue su curso sola.
    setRecording(false);

    const mimeType = recorder.mimeType || "video/webm";
    const blob = new Blob(chunksRef.current, { type: mimeType });
    const ext = mimeType.includes("mp4") ? "mp4" : "webm";
    runPrediction(blob, `clip.${ext}`, audioBlob);
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
      startAudioCapture();
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

  // --- Modo en vivo: segmentos cortos encadenados, prendido por el propio
  // usuario mientras practica, para no tener que tocar "grabar" después de
  // cada intento (ver constantes LIVE_* arriba). ---
  const startLiveSegment = useCallback(() => {
    const stream = streamRef.current;
    if (!stream || !liveModeRef.current) return;

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
      startAudioCapture();
      setRecording(true);

      autoStopTimerRef.current = setTimeout(
        () => finishLiveSegmentRef.current(),
        LIVE_SEGMENT_MS
      );
    } catch (e) {
      setLastError(`No se pudo grabar el segmento: ${e.message || e}`);
      liveModeRef.current = false;
      setLiveMode(false);
    }
  }, []);

  const finishLiveSegment = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;

    const stopped = new Promise((resolve) => {
      recorder.onstop = resolve;
    });
    recorder.stop();
    await stopped;
    const audioBlob = stopAudioCapture();
    setRecording(false);

    const mimeType = recorder.mimeType || "video/webm";
    const blob = new Blob(chunksRef.current, { type: mimeType });
    const ext = mimeType.includes("mp4") ? "mp4" : "webm";

    // Encadena el siguiente segmento recién cuando ESTA predicción termina
    // + una pausa corta -- mandar segmentos pegados uno atrás de otro sin
    // pausa es lo que dispara cuelgues intermitentes del driver de GPU (ver
    // el comentario grande sobre esto en server/main.py).
    const jobPromise = runPrediction(blob, `segment.${ext}`, audioBlob);
    jobPromise.finally(() => {
      if (liveModeRef.current) {
        liveCooldownTimerRef.current = setTimeout(() => {
          liveCooldownTimerRef.current = null;
          startLiveSegment();
        }, LIVE_COOLDOWN_MS);
      }
    });
  }, [runPrediction, startLiveSegment]);

  useEffect(() => {
    finishLiveSegmentRef.current = finishLiveSegment;
  }, [finishLiveSegment]);

  function toggleLiveMode() {
    if (liveMode) {
      liveModeRef.current = false;
      setLiveMode(false);
      if (autoStopTimerRef.current) {
        clearTimeout(autoStopTimerRef.current);
        autoStopTimerRef.current = null;
      }
      if (liveCooldownTimerRef.current) {
        clearTimeout(liveCooldownTimerRef.current);
        liveCooldownTimerRef.current = null;
      }
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.onstop = null; // se apaga sin mandar el segmento a medio grabar
        recorder.stop();
      }
      stopAudioCapture(); // corta y descarta -- este segmento no se manda
      setRecording(false);
    } else {
      setLastResult(null);
      setLastError(null);
      liveModeRef.current = true;
      setLiveMode(true);
      startLiveSegment();
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
            {onBack && (
              <button className="icon-button" onClick={onBack} title="Volver">
                ←
              </button>
            )}
            <span className="brand-mascot">🗣️</span>
            <span>FonoKids</span>
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
          {!initialWord && targetWords.length > 0 && (
            <div className="word-picker">
              {targetWords.map((w) => (
                <button
                  key={w}
                  className={`word-chip ${targetWord === w ? "selected" : ""}`}
                  onClick={() => setTargetWord(w)}
                  disabled={recording || liveMode}
                >
                  <span className="word-chip-icon">{iconFor(w)}</span>
                  {w.replaceAll("_", " ")}
                </button>
              ))}
            </div>
          )}

          {targetWord && (
            <div className="target-word-banner">
              Decí: <span className="target-word-banner-word">{iconFor(targetWord)} {targetWord.replaceAll("_", " ")}</span>
            </div>
          )}

          <div className={`camera-frame ${recording ? "is-recording" : ""} ${liveMode ? "is-live" : ""}`}>
            {cameraError ? (
              <div className="center-message">{cameraError}</div>
            ) : (
              <>
                <video ref={videoRef} className="camera-preview" autoPlay playsInline muted />
                <canvas ref={canvasRef} className="camera-overlay" />
              </>
            )}
            {liveMode && <div className="live-badge">🔴 EN VIVO</div>}
          </div>

          <div className="stage-hint">
            {retryingPredict
              ? "⏳ El servidor se colgó, esperando a que vuelva para reintentar solo..."
              : lastError
              ? `⚠ ${lastError}`
              : liveMode
              ? `Modo en vivo -- segmento cada ${LIVE_SEGMENT_MS / 1000}s, seguí hablando`
              : recording
              ? "Grabando... tocá para terminar"
              : processingCount > 0
              ? `Analizando ${processingCount} intento(s)...`
              : "Decí la palabra y tocá el botón para grabar"}
          </div>

          <div className="stage-controls">
            <button
              className={`live-toggle-button ${liveMode ? "active" : ""}`}
              onClick={toggleLiveMode}
              disabled={!!cameraError || (recording && !liveMode)}
              title="Modo en vivo: graba y evalúa en segmentos cortos automáticos, mientras seguís hablando"
            >
              {liveMode ? "⏹ Parar" : "🔴 En vivo"}
            </button>
            <div className="record-button-wrap">
              <button
                className={`record-button ${recording ? "recording" : ""}`}
                onClick={liveMode ? toggleLiveMode : toggleRecording}
                disabled={!!cameraError || (liveMode && recording)}
              >
                <span className={recording ? "square" : "circle"} />
              </button>
              {processingCount > 0 && <span className="processing-badge">{processingCount}</span>}
            </div>
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
        <ResultOverlay
          result={lastResult}
          onClose={() => setLastResult(null)}
          autoCloseMs={liveMode ? LIVE_RESULT_AUTOCLOSE_MS : null}
        />
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
/// puede tocar afuera para cerrarla antes. En modo en vivo (autoCloseMs)
/// se cierra sola después de un rato, para no cortar la práctica continua
/// esperando un tap que en ese modo nadie va a hacer.
function ResultOverlay({ result, onClose, autoCloseMs }) {
  useEffect(() => {
    if (!autoCloseMs) return undefined;
    const timer = setTimeout(onClose, autoCloseMs);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, autoCloseMs]);

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
  // Pedido explícito: cuando está MAL, no mostrar un "% de seguridad" (eso
  // solo tiene sentido para el "bien dicho") -- en cambio, mostrar lo que
  // realmente se escuchó (sonido_reconocido, ver audio_pronunciation.py),
  // así el chico/padre ve "sonó como 'pelo'" en vez de un número sin
  // contexto que además, para casos como "pelo" vs "perro" (1 solo fonema
  // distinto, la distancia de edición los toma como "misma palabra"), podía
  // mostrar "perro" como si eso fuera lo que se dijo.
  const sonidoReconocido = result.audio?.sonido_reconocido;
  return (
    <div className="result-overlay" onClick={onClose}>
      <div className={`result-overlay-card ${result.correcta ? "success" : "retry"}`} onClick={(e) => e.stopPropagation()}>
        <div className="result-overlay-emoji">{result.correcta ? "🎉" : "🔁"}</div>
        <div className="result-overlay-word">{palabra}</div>
        {result.correcta ? (
          <div className="result-overlay-message">¡Muy bien dicho! -- {pct}% de seguridad</div>
        ) : sonidoReconocido ? (
          <div className="result-overlay-message">
            Sonó como "{sonidoReconocido}" -- practiquemos "{palabra}" de nuevo
          </div>
        ) : (
          <div className="result-overlay-message">Casi... practiquemos de nuevo</div>
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
