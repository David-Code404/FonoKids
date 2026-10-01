// Cliente HTTP para server/main.py -- equivalente web de api_client.dart.
const STORAGE_KEY = "speakshadow_server_url";
// server/main.py corre en la MISMA máquina que sirve esta página ("npm run
// dev" y "python server/main.py" en la misma PC, ambos en localhost).
export const DEFAULT_URL = `${window.location.protocol}//${window.location.hostname}:8000`;

export class ApiError extends Error {}

function normalize(url) {
  const u = url.trim();
  return u.endsWith("/") ? u.slice(0, -1) : u;
}

export function getServerUrl() {
  return localStorage.getItem(STORAGE_KEY) || DEFAULT_URL;
}

/// URL de la foto real (frame del medio del clip) para una captura --
/// GET /dataset/thumbnail en server/main.py. Puede devolver 404 si el clip
/// es viejo (dataset de entrenamiento) o no tiene thumbnail_file -- la UI
/// debe manejar el error de carga y caer al placeholder.
export function thumbnailUrl(baseUrl, word, filename) {
  const params = new URLSearchParams({ word, filename });
  return `${normalize(baseUrl)}/dataset/thumbnail?${params}`;
}

export function setServerUrl(url) {
  localStorage.setItem(STORAGE_KEY, url.trim());
}

async function withTimeout(promise, ms, timeoutMessage) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await promise(controller.signal);
  } catch (e) {
    if (e.name === "AbortError") {
      throw new ApiError(timeoutMessage);
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export async function checkHealth(baseUrl) {
  try {
    const res = await withTimeout(
      (signal) => fetch(`${normalize(baseUrl)}/health`, { signal }),
      5000,
      "timeout"
    );
    return res.ok;
  } catch {
    return false;
  }
}

/// Igual que checkHealth pero devuelve el cuerpo entero (incluye "classes",
/// la lista de clases que reconoce el modelo cargado) en vez de solo
/// true/false -- lo usa el selector de palabra objetivo en Practicar.
/// null si el server no responde (mismo criterio que checkHealth).
export async function getHealth(baseUrl) {
  try {
    const res = await withTimeout(
      (signal) => fetch(`${normalize(baseUrl)}/health`, { signal }),
      5000,
      "timeout"
    );
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/// Secuencia real de puntos de labios (20 puntos x,y por frame) de un clip
/// "<palabra>_correcto" -- para animar la boquita de referencia. Null si
/// todavía no hay ningún clip correcto grabado de esa palabra.
export async function getReferenceLandmarks(baseUrl, palabra) {
  try {
    const res = await withTimeout(
      (signal) => fetch(`${normalize(baseUrl)}/reference/${encodeURIComponent(palabra)}`, { signal }),
      20000,
      "timeout"
    );
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/// De la lista de clases del modelo ("perro_correcto", "perro_incorrecto_
/// lambdacismo", ...) saca las palabras objetivo ÚNICAS ("perro", ...),
/// mismo criterio que _parse_clase_predicha() en server/main.py -- corta
/// en "_correcto" o en "_incorrecto" (lo que aparezca primero).
export function wordsFromClasses(classNames) {
  if (!Array.isArray(classNames)) return [];
  const words = new Set();
  for (const c of classNames) {
    const idx = c.endsWith("_correcto") ? c.length - "_correcto".length : c.indexOf("_incorrecto");
    words.add(idx > 0 ? c.slice(0, idx) : c);
  }
  return [...words].sort();
}

// 20s en vez de 5s -- estas consultas son livianas (listar archivos), pero
// comparten servidor con /predict: si justo hay una predicción corriendo,
// pueden demorar un poco más de lo normal en contestar. Con solo 5s
// cualquier pestañeo hacía que la app cayera a datos de muestra y pareciera
// que "cambiaban solas" entre reales y de muestra.
const DATASET_TIMEOUT_MS = 20000;

export async function getDatasetStats(baseUrl) {
  const res = await withTimeout(
    (signal) => fetch(`${normalize(baseUrl)}/dataset/stats`, { signal }),
    DATASET_TIMEOUT_MS,
    "El servidor tardó demasiado en responder."
  ).catch((e) => {
    throw new ApiError(e.message || `No se pudo conectar al servidor (${baseUrl}).`);
  });
  if (!res.ok) throw new ApiError(`El servidor respondió con error ${res.status}.`);
  return res.json();
}

export async function getRecordings(baseUrl) {
  const res = await withTimeout(
    (signal) => fetch(`${normalize(baseUrl)}/dataset/recordings`, { signal }),
    DATASET_TIMEOUT_MS,
    "El servidor tardó demasiado en responder."
  ).catch((e) => {
    throw new ApiError(e.message || `No se pudo conectar al servidor (${baseUrl}).`);
  });
  if (!res.ok) throw new ApiError(`El servidor respondió con error ${res.status}.`);
  return res.json();
}

async function waitForServerBack(baseUrl, maxWaitMs = 60000) {
  // El servidor se reinicia solo cuando /predict se cuelga (ver server/
  // main.py, PREDICT_HARD_TIMEOUT_S) -- típicamente vuelve a responder en
  // 20-30s. Sondeamos /health hasta que vuelva, en vez de reintentar a
  // ciegas contra un servidor que todavía se está reiniciando.
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    if (await checkHealth(baseUrl)) return true;
    await new Promise((r) => setTimeout(r, 2500));
  }
  return false;
}

async function predictOnce(baseUrl, videoBlob, filename, audioBlob, palabra) {
  const uri = `${normalize(baseUrl)}/predict`;
  const form = new FormData();
  // El video es OPCIONAL -- modo "solo voz" (sin cámara, ver
  // CameraConsentGate en HomeScreen.jsx): no se manda "file" en absoluto, y
  // el servidor decide el veredicto solo con el audio.
  if (videoBlob) form.append("file", videoBlob, filename);
  // Audio y palabra son OPCIONALES -- si no hay micrófono (o el navegador
  // lo bloqueó), el servidor sigue funcionando solo con el video, igual que
  // antes de agregar el análisis de sonido.
  if (audioBlob) form.append("audio", audioBlob, "clip.wav");
  if (palabra) form.append("palabra", palabra);

  let response;
  try {
    response = await withTimeout(
      (signal) => fetch(uri, { method: "POST", body: form, signal }),
      150000,
      "El servidor tardó demasiado en responder. Puede estar sobrecargado " +
        "(GPU con poca memoria) -- probá de nuevo en unos segundos."
    );
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(
      `No se pudo conectar al servidor (${baseUrl}). Revisá que esté prendido (python server/main.py).`
    );
  }

  if (!response.ok) {
    // El servidor manda un "detail" específico y útil (ej. "se está
    // reiniciando solo, probá de nuevo en unos segundos" en un 504) --
    // antes se ignoraba y siempre se mostraba un mensaje genérico. Mejor
    // mostrar el real cuando existe.
    let detail = null;
    try {
      const body = await response.json();
      detail = body?.detail || null;
    } catch {
      // Respuesta sin JSON (ej. error crudo del servidor web) -- seguimos
      // con el mensaje genérico de abajo.
    }
    const err = new ApiError(
      detail || `El servidor respondió con error ${response.status}. Revisá que el modelo esté cargado ahí.`
    );
    err.status = response.status;
    throw err;
  }

  return response.json();
}

/// Envía el clip grabado (Blob) al servidor y devuelve la predicción.
//
// Reintento automático ante un cuelgue de GPU: el servidor a veces se
// cuelga por un problema intermitente del driver de NVIDIA/CUDA (no del
// código ni del clip -- confirmado probando el MISMO clip varias veces:
// unas anda, otras se cuelga) y se mata/reinicia solo en ~90s (ver
// PREDICT_HARD_TIMEOUT_S en server/main.py). Antes, cuando pasaba eso, el
// usuario tenía que darse cuenta del error y volver a grabar a mano. Ahora,
// si la respuesta es justo ese 504 de "me estoy reiniciando", esperamos a
// que el servidor vuelva a responder /health y reintentamos el MISMO clip
// una sola vez automáticamente -- en la mayoría de los casos ya funciona
// en el segundo intento, sin que el usuario tenga que hacer nada.
/// Evalúa UNA palabra entrenada dentro de una FRASE completa (ver
/// FrasesScreen.jsx y server/main.py POST /predict_frase) -- distinto de
/// predict() de abajo, que evalúa la palabra sola con video+audio. Acá
/// solo se manda audio, nunca video (el modo frase es siempre "solo voz").
export async function predictFrase(baseUrl, audioBlob, frase, palabra) {
  const uri = `${normalize(baseUrl)}/predict_frase`;
  const form = new FormData();
  form.append("audio", audioBlob, "frase.wav");
  form.append("frase", frase);
  form.append("palabra", palabra);

  let response;
  try {
    response = await withTimeout(
      (signal) => fetch(uri, { method: "POST", body: form, signal }),
      60000,
      "El servidor tardó demasiado en responder -- probá de nuevo."
    );
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(
      `No se pudo conectar al servidor (${baseUrl}). Revisá que esté prendido.`
    );
  }

  if (!response.ok) {
    let detail = null;
    try {
      const body = await response.json();
      detail = body?.detail || null;
    } catch {
      // sin JSON -- mensaje genérico de abajo
    }
    throw new ApiError(detail || `El servidor respondió con error ${response.status}.`);
  }

  return response.json();
}

export async function predict(
  baseUrl, videoBlob, filename = "clip.webm", onRetrying = null, audioBlob = null, palabra = null
) {
  try {
    return await predictOnce(baseUrl, videoBlob, filename, audioBlob, palabra);
  } catch (e) {
    if (e.status !== 504) throw e;

    if (onRetrying) onRetrying();
    const back = await waitForServerBack(baseUrl);
    if (!back) {
      throw new ApiError(
        "El servidor se colgó y todavía no volvió a responder -- probá de nuevo en un ratito."
      );
    }
    return await predictOnce(baseUrl, videoBlob, filename, audioBlob, palabra);
  }
}
