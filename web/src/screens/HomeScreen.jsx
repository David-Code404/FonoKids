import { useCallback, useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import {
  checkHealth, getHealth, getServerUrl, predict,
  setServerUrl as saveServerUrl, wordsFromClasses,
} from "../api.js";
import { iconFor } from "../wordIcons.js";
import { tipoErrorInfo } from "../errorTips.js";
import SettingsDialog from "../components/SettingsDialog.jsx";
import "./HomeScreen.css";

// Tope de duración de una toma -- coincide con MAX_FRAMES_BUFFER de
// probar_modelo.py (~150 frames a 25fps). Si el usuario se olvida de tocar
// para detener, esto corta solo en vez de mandar un clip cada vez más largo.
const MAX_RECORDING_MS = 6000;
const CONNECTION_REFRESH_MS = 20000;
// Pedido explícito: el chico decide si quiere prender la cámara, no se le
// pide permiso al navegador de entrada apenas abre la pantalla -- se
// recuerda en este navegador (localStorage) para no preguntar cada vez que
// practica, pero siempre se puede "cambiar de opinión" desde la pantalla.
const CAMERA_CONSENT_KEY = "fonokids_camera_consent";

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
export default function HomeScreen({
  initialWord = null, promptText = null, promptIcon = null, onBack = null,
  // Apartado separado "solo audio" (ver LearningPathScreen) -- acá NUNCA se
  // pide ni se menciona la cámara, ni siquiera como opción. Arranca
  // directo pidiendo el micrófono apenas se monta la pantalla.
  forceAudioOnly = false,
  // Callback opcional -- se llama con el `result` completo del servidor
  // cada vez que una predicción vuelve válida (result.valid). Lo usa
  // DiagnosticoScreen para saber en qué le fue a cada palabra sin tener
  // que duplicar toda la lógica de grabar/mandar/mostrar overlay.
  onResult = null,
}) {
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
  // Ref espejo de targetWord -- se lee desde dentro de closures async
  // (runPrediction) donde un closure sobre el state de React podría quedar
  // desactualizado entre renders.
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
  // Stream de SOLO MICRÓFONO -- para cuando el chico elige practicar sin
  // cámara (ver acceptAudioOnly más abajo). Separado de streamRef (que es
  // el de cámara+mic juntos) porque son dos pedidos de permiso distintos.
  const audioOnlyStreamRef = useRef(null);
  const audioOnlyRecordingRef = useRef(false);

  const [cameraError, setCameraError] = useState(null);
  // null = todavía no contestó, true = dijo que sí, false = dijo que no.
  const [cameraConsent, setCameraConsent] = useState(() => {
    try {
      return localStorage.getItem(CAMERA_CONSENT_KEY) === "yes" ? true : null;
    } catch {
      return null;
    }
  });
  // Modo de práctica SIN cámara, solo con el micrófono -- el modelo de
  // audio solo (~99% de accuracy real) puede decidir el veredicto sin
  // necesitar video, ver server/main.py (_run_audio_only_pipeline).
  const [audioOnlyMode, setAudioOnlyMode] = useState(false);
  const [recording, setRecording] = useState(false);
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
    // No se pide cámara/micrófono hasta que el chico dice que sí -- ver
    // CameraConsentGate más abajo.
    if (cameraConsent !== true) return;

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
  }, [cameraConsent]);

  function acceptCameraConsent() {
    try {
      localStorage.setItem(CAMERA_CONSENT_KEY, "yes");
    } catch {
      // localStorage puede fallar (modo privado, storage bloqueado) -- no
      // pasa nada, simplemente va a volver a preguntar la próxima vez.
    }
    setCameraConsent(true);
  }

  // El chico eligió NO usar cámara -- practica solo con el micrófono, el
  // clasificador de audio entrenado decide el veredicto solo (ver
  // AUDIO_MIN_CONFIDENCE en server/main.py).
  async function startAudioOnlyPractice() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioOnlyStreamRef.current = stream;
      setAudioOnlyMode(true);
      setCameraError(null);
    } catch (e) {
      setCameraError(`No se pudo usar el micrófono: ${e.message || e}`);
    }
  }

  useEffect(() => {
    return () => {
      audioOnlyStreamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // Apartado "solo audio" -- pide el micrófono apenas se monta la pantalla,
  // sin preguntar nada y sin mostrar la cámara en ningún momento (a
  // diferencia del modo normal, que primero pregunta si prender la cámara).
  useEffect(() => {
    if (!forceAudioOnly) return;
    startAudioOnlyPractice();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [forceAudioOnly]);

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
    // BUG real detectado: sin "initialWord" en las dependencias, este
    // efecto solo corría cuando cambiaba la conexión -- si LearningPathScreen
    // reutiliza el mismo HomeScreen para pasar de una palabra a otra (ej.
    // "gorra_medias" -> "globo") sin recargar la página, el título en
    // pantalla se actualizaba (viene de promptText) pero targetWord (lo que
    // de verdad se manda al servidor en cada intento) quedaba pegado en la
    // palabra anterior -- el chico veía "GLOBO" pero el server evaluaba
    // contra "gorra_medias", así que nunca podía dar bien.
  }, [serverUrl, connState, initialWord]);

  useEffect(() => {
    targetWordRef.current = targetWord;
  }, [targetWord]);

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
    // Pedido explícito: no mostrar/guardar fotos de la cara del chico en el
    // historial -- solo ícono genérico, sin importar si el servidor guardó
    // un clip de práctica.
    const entry = { id, palabra, prob, correcta, time: new Date() };
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
          if (result.valid && onResult) onResult(result);
          if (result.valid && result.palabra) {
            // Mismo criterio que en ResultOverlay: la racha tiene que
            // mostrar lo que se le pidió decir en ESTE paso (ej. "PER"),
            // no la familia que devolvió el servidor -- si estaba
            // practicando "per" y le salió mal, no tiene sentido que la
            // tarjeta del historial diga "perro".
            registerAttempt(
              promptText || result.palabra, result.prob ?? 0,
              result.capture_file, result.class_name, result.correcta
            );
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
  // grabando solo video, como antes de agregar esto. En modo solo-audio
  // (sourceStream) usa el stream de solo-mic en vez del de cámara+mic.
  function startAudioCapture(sourceStream) {
    audioChunksRef.current = [];
    const stream = sourceStream || streamRef.current;
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
    // Modo solo-audio (sin cámara): no hay MediaRecorder de video, solo se
    // corta la captura de audio y se manda sin video -- el servidor decide
    // el veredicto solo con el clasificador de audio entrenado (ver
    // _run_audio_only_pipeline en server/main.py).
    if (audioOnlyMode) {
      if (!audioOnlyRecordingRef.current) return;
      if (autoStopTimerRef.current) {
        clearTimeout(autoStopTimerRef.current);
        autoStopTimerRef.current = null;
      }
      audioOnlyRecordingRef.current = false;
      const audioBlob = stopAudioCapture();
      setRecording(false);
      runPrediction(null, null, audioBlob);
      return;
    }

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
  }, [runPrediction, audioOnlyMode]);

  const startRecording = useCallback(() => {
    setLastResult(null);
    setLastError(null);

    if (audioOnlyMode) {
      const stream = audioOnlyStreamRef.current;
      if (!stream) return;
      audioOnlyRecordingRef.current = true;
      startAudioCapture(stream);
      setRecording(true);
      autoStopTimerRef.current = setTimeout(() => {
        stopAndPredict();
      }, MAX_RECORDING_MS);
      return;
    }

    const stream = streamRef.current;
    if (!stream) return;

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
  }, [stopAndPredict, audioOnlyMode]);

  function toggleRecording() {
    // Pedido explícito: un intento por vez -- el botón queda deshabilitado
    // mientras el servidor todavía está analizando la toma anterior (ver
    // "disabled" en el botón de grabar más abajo), así no se pisan varias
    // predicciones encadenadas.
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

  const connLabel = { ok: "¡Listo! 🎉", fail: "Sin conexión", unknown: "Conectando..." }[connState];
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

        {!forceAudioOnly && cameraConsent !== true && !audioOnlyMode ? (
          <CameraConsentGate
            declined={cameraConsent === false}
            error={cameraError}
            onAccept={acceptCameraConsent}
            onDecline={() => setCameraConsent(false)}
            onAudioOnly={startAudioOnlyPractice}
          />
        ) : (
        <div className="practice-stage">
          {!initialWord && targetWords.length > 0 && (
            <div className="word-picker">
              {targetWords.map((w) => (
                <button
                  key={w}
                  className={`word-chip ${targetWord === w ? "selected" : ""}`}
                  onClick={() => setTargetWord(w)}
                  disabled={recording}
                >
                  <span className="word-chip-icon">{iconFor(w)}</span>
                  {w.replaceAll("_", " ")}
                </button>
              ))}
            </div>
          )}

          {(promptText || targetWord) && (
            <div className="target-word-banner">
              Decí: <span className="target-word-banner-word">
                {iconFor(promptIcon || targetWord)} {promptText || targetWord.replaceAll("_", " ")}
              </span>
            </div>
          )}

          {audioOnlyMode || forceAudioOnly ? (
            <div className="audio-only-stage">
              {cameraError ? (
                <div className="audio-only-error">{cameraError}</div>
              ) : !audioOnlyMode ? (
                <>
                  <div className="audio-only-orb">
                    <span className="audio-only-icon">🎤</span>
                  </div>
                  <span className="audio-only-label">Activando el micrófono...</span>
                </>
              ) : (
                <>
                  <div className={`audio-only-orb ${recording ? "is-recording" : ""}`}>
                    {recording && (
                      <>
                        <span className="audio-only-pulse audio-only-pulse-1" />
                        <span className="audio-only-pulse audio-only-pulse-2" />
                      </>
                    )}
                    <span className="audio-only-icon">🎤</span>
                  </div>
                  <span className="audio-only-label">
                    {recording ? "Escuchando..." : "Practicando solo con la voz"}
                  </span>
                </>
              )}
            </div>
          ) : (
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
          )}

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

          <div className="stage-controls">
            <div className="record-button-wrap">
              <button
                className={`record-button ${recording ? "recording" : ""}`}
                onClick={toggleRecording}
                disabled={!!cameraError || (processingCount > 0 && !recording)}
              >
                <span className={recording ? "square" : "circle"} />
              </button>
              {processingCount > 0 && <span className="processing-badge">{processingCount}</span>}
            </div>
          </div>
        </div>
        )}
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
          promptText={promptText}
          onClose={() => setLastResult(null)}
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
/// puede tocar afuera para cerrarla antes.
function ResultOverlay({ result, promptText, onClose }) {
  if (!result.valid) {
    // Pedido explícito: nunca mostrarle al chico el motivo técnico crudo
    // que manda el servidor (ej. "cara detectada solo en 0/75 frames --
    // encuadre malo") -- un mensaje simple y accionable alcanza.
    return (
      <div className="result-overlay" onClick={onClose}>
        <div className="result-overlay-card" onClick={(e) => e.stopPropagation()}>
          <div className="result-overlay-emoji">🙈</div>
          <div className="result-overlay-message">No te vi bien la carita -- ¡probemos de nuevo!</div>
          <button className="result-overlay-button neutral" onClick={onClose}>
            Dale, de nuevo
          </button>
        </div>
      </div>
    );
  }

  // Mostramos lo que le PEDIMOS que diga (promptText, ej. "RR"), no lo que
  // el servidor reconoció (result.palabra) -- si estaba en el paso "RR" y
  // el modelo entendió otra cosa (ej. la palabra completa), mostrar
  // "Perro" como título confunde: parece que evaluamos otra palabra, no
  // que rechazamos su intento por no ser el sonido que tocaba en este paso.
  const palabra = promptText || (result.palabra || "?").replaceAll("_", " ");

  // La decisión (bien/mal dicha) es del servidor (softmax + margen sobre
  // clases "<palabra>_correcto"/"<palabra>_incorrecto", ver server/main.py)
  // -- no se recalcula acá.
  if (!result.confianza_suficiente) {
    return (
      <div className="result-overlay" onClick={onClose}>
        <div className="result-overlay-card" onClick={(e) => e.stopPropagation()}>
          <div className="result-overlay-emoji">🤔</div>
          <div className="result-overlay-message">No estoy seguro de lo que dijiste -- ¡repetí la toma!</div>
          <button className="result-overlay-button neutral" onClick={onClose}>
            Dale, de nuevo
          </button>
        </div>
      </div>
    );
  }

  // Pedido explícito: nada de porcentajes en pantalla -- a un chico no le
  // interesa "82% de seguridad", le interesa si lo hizo bien o no. El
  // umbral de confianza (MIN_CONFIDENCE_PROB/MARGIN en server/main.py) ya
  // decide todo eso antes de llegar acá -- si el servidor devuelve
  // "correcta", ya pasó ese umbral, no hace falta mostrar el número.
  //
  // Cuando está MAL, en vez de un número mostramos, si el servidor
  // identificó el tipo de error (tipo_error: lambdacismo, dentalización,
  // omisión), un consejo concreto de qué probar distinto -- pedido
  // explícito: no alcanza con "está mal", el chico tiene que ver DÓNDE le
  // salió mal para poder corregirlo, no solo que se repita.
  //
  // Ojo: NUNCA mostramos el "sonido_reconocido" crudo (ver
  // audio_pronunciation.py) en pantalla -- es una transcripción fonética
  // ("krrabo", etc.), no una palabra real, y muchos chicos todavía no
  // pueden leer bien -- mejor un mensaje simple y un consejo claro.
  const errorInfo = !result.correcta && result.tipo_error ? tipoErrorInfo(result.tipo_error) : null;
  return (
    <div className="result-overlay" onClick={onClose}>
      <div className={`result-overlay-card ${result.correcta ? "success" : "retry"}`} onClick={(e) => e.stopPropagation()}>
        <div className="result-overlay-emoji">{result.correcta ? "🎉" : (result.palabra_distinta ? "🙃" : (errorInfo?.icon || "🔁"))}</div>
        <div className="result-overlay-word">{palabra}</div>
        {result.correcta ? (
          <div className="result-overlay-message">¡Muy bien dicho!</div>
        ) : result.palabra_distinta ? (
          <div className="result-overlay-message">
            Esa no es la palabra de este paso -- ¡probemos decir "{palabra}"!
          </div>
        ) : (
          <div className="result-overlay-message">😢 ¡Casi! Practiquemos "{palabra}" de nuevo</div>
        )}
        {errorInfo && !result.palabra_distinta && (
          <div className="result-overlay-tip">
            <span className="result-overlay-tip-label">{errorInfo.label}</span>
            <span className="result-overlay-tip-text">{errorInfo.tip}</span>
          </div>
        )}
        <button className={`result-overlay-button ${result.correcta ? "success" : "retry"}`} onClick={onClose}>
          {result.correcta ? "¡Genial! 🌟" : "Intentar de nuevo"}
        </button>
      </div>
    </div>
  );
}

/// Ícono genérico del intento -- pedido explícito: nunca mostrar una foto
/// real de la cara del chico acá, ni aunque el servidor haya guardado un
/// clip de práctica.
function AttemptAvatar() {
  return <div className="capture-avatar">🗣</div>;
}

/// Pantalla que decide SI se prende la cámara -- pedido explícito: que sea
/// el chico quien elija, en vez de pedirle permiso al navegador apenas abre
/// la pantalla de práctica sin preguntar nada primero.
function CameraConsentGate({ declined, error, onAccept, onDecline, onAudioOnly }) {
  return (
    <div className="camera-consent-gate">
      <div className="camera-consent-icon">{declined ? "🎤" : "📷"}</div>
      {error && (
        <div className="camera-consent-text camera-consent-error">
          No pudimos prender el micrófono -- fijate que le hayas dado permiso a la app en tu navegador.
        </div>
      )}
      {declined ? (
        <>
          <div className="camera-consent-title">¡No hay problema!</div>
          <div className="camera-consent-text">
            Podés seguir practicando solo con tu voz, o prender la cámara cuando quieras.
          </div>
          <div className="camera-consent-buttons">
            <button className="camera-consent-button primary" onClick={onAudioOnly}>
              Practicar solo con la voz 🎤
            </button>
            <button className="camera-consent-button ghost" onClick={onAccept}>
              Mejor prendo la cámara
            </button>
          </div>
        </>
      ) : (
        <>
          <div className="camera-consent-title">¿Prendemos la cámara?</div>
          <div className="camera-consent-text">
            La vamos a usar para ver cómo movés la boca mientras practicás -- vos decidís.
          </div>
          <div className="camera-consent-buttons">
            <button className="camera-consent-button primary" onClick={onAccept}>
              Sí, dale 🎥
            </button>
            <button className="camera-consent-button ghost" onClick={onDecline}>
              Ahora no
            </button>
          </div>
        </>
      )}
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
              <AttemptAvatar />
              <div className="capture-info">
                <div className="capture-row">
                  <span className="capture-word">{entry.palabra.replaceAll("_", " ")}</span>
                  <button className="dismiss-button" onClick={() => onDismiss(entry.id)}>
                    ✕
                  </button>
                </div>
                <div className="capture-sub">
                  {entry.time.getHours().toString().padStart(2, "0")}:
                  {entry.time.getMinutes().toString().padStart(2, "0")}
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
