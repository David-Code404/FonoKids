"""
server/main.py
---------------
API HTTP para la app de Flutter. Recibe un video corto (grabado desde el
celular), corre EXACTAMENTE el mismo pipeline que probar_modelo.py
(HRNet -> landmarks de labios -> TCN + Conformer) y devuelve la palabra
predicha en JSON.

La detección de landmarks corre en BATCH (todo el clip al GPU de una sola
vez, ver detect_landmarks_batch en extraer_landmarks_npy.py) en vez de
frame por frame -- es lo que hace que /predict responda en segundos y no
en 15-20s con HRNet.

Uso:
    python server/main.py
    (por defecto escucha en 0.0.0.0:8000 -- accesible desde el celular
    si está en la misma red WiFi que esta PC, usando la IP local de la PC)
"""
import asyncio
import os
import queue
import sys
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from train_landmarks_transformer import LipReadingConformer, make_padding_mask  # noqa: E402
from extraer_landmarks_npy import (  # noqa: E402
    build_landmarker,
    detect_landmarks_batch,
    normalize_lip_landmarks,
    fill_missing_frames,
    smooth_positions,
    add_dynamics,
    is_outlier,
)

DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset_pt")
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mejor_modelo_landmarks_conformer.pth")
GOAL_PER_WORD = 500

MIN_FRAMES_VALID = 15           # menos que esto = probablemente no se dijo la frase completa
MIN_DETECTION_RATE_VALID = 0.6  # menos del 60% de frames con cara detectada = toma mala

# NO se recortan ni saltean frames -- igual que probar_modelo.py/extraer_
# landmarks_npy.py ("CONTINUIDAD TEMPORAL ESTRICTA"). Antes acá había un
# tope (MAX_PIPELINE_FRAMES) que cortaba clips largos para evitar el
# cuelgue de GPU, pero eso rompía la continuidad temporal con la que se
# entrenó el modelo y empeoraba la predicción. El cuelgue real ya se
# arregla con el resize (MAX_FRAME_WIDTH) + el chunking de sub-lotes en
# detect_landmarks_batch (ver extraer_landmarks_npy.py) -- con eso ya no
# hace falta cortar el clip.

# Ancho máximo de cada frame ANTES de mandarlo a HRNet -- la cámara del
# navegador graba en Full HD (1920px), muchísimo más de lo necesario para
# detectar una cara. 640px es lo mismo que usa grabar_video_continuo.py /
# probar_modelo.py (FRAME_SIZE), donde nunca se vio este cuelgue.
MAX_FRAME_WIDTH = 640

# Igual que probar_modelo.py: el modelo SIEMPRE elige una de las clases
# entrenadas (no sabe decir "no sé"), así que esto es lo que distingue
# "sí es una frase de riesgo conocida" de "no es nada, no hacer caso".
MIN_RISK_PROB = 0.90      # la clase top tiene que superar esto
MIN_RISK_MARGIN = 0.30    # y sacarle esta diferencia mínima a la 2da opción
# (sincronizado con probar_modelo.py -- se subió de 0.40/0.15 porque con
# umbrales bajos alcanzaba con un 40% de confianza en CUALQUIER frase
# entrenada, incluida una neutra, para marcar riesgo por error.)

# Las ÚNICAS clases que cuentan como "frase de riesgo" -- el resto de las
# clases del modelo (las 20 frases neutras tipo "hola"/"gracias", ver
# scripts/generar_sintetico.py) están ahí para que el modelo tenga algo
# real que predecir cuando NO es bullying, pero nunca tienen que disparar
# una alerta aunque salgan con confianza alta. Antes esto no se chequeaba
# -- la decisión de riesgo miraba SOLO la confianza del softmax, sin
# importar qué palabra era, así que una frase neutra predicha con
# confianza igual se marcaba como riesgo por error.
RISK_WORDS = frozenset({
    "asqueroso", "camba_de_mierda", "ciego_de_mierda", "colla_y_mierda",
    "de_esta_no_te_salvas", "enano", "engendro", "eres_una_rata", "estorbo",
    "feo", "fracasado", "fracasado_de_mierda", "gordo", "gordo_asqueroso",
    "idiota", "inservible", "maldito_colla_de_mierda", "estúpido",
    "imbécil", "inútil",
})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)

app = FastAPI(title="SpeakShadow API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global: modelo y landmarker se cargan UNA sola vez al arrancar,
# no en cada request (cargar el modelo por request sería lentísimo).
_state = {"model": None, "landmarker": None, "class_names": None, "max_frames": None}

# =====================================================================
# HILO DEDICADO DE GPU -- ver por qué esto existe en el comentario largo
# de más abajo. TL;DR: antes cada /predict corría vía run_in_threadpool
# (el pool genérico de Starlette/AnyIO), que puede repartir cada request
# en un hilo de sistema operativo DISTINTO. En probar_modelo.py, en
# cambio, TODO (cargar el modelo Y cada predicción) corre siempre en el
# mismo hilo (el principal) -- y ahí nunca se cuelga, ni siquiera con
# tomas largas. La diferencia real entre "funciona en probar_modelo.py"
# y "se cuelga en el servidor" parece ser justo esa: CUDA en Windows es
# conocido por dar problemas cuando el contexto de GPU se toca desde
# hilos distintos a lo largo de la vida del proceso. Acá se fuerza que
# absolutamente todo el trabajo de GPU (carga del modelo, precalentamiento,
# y cada predicción) pase SIEMPRE por el mismo único hilo, de punta a
# punta, igual que probar_modelo.py.
# =====================================================================
_gpu_queue = queue.Queue()


def _gpu_worker_loop():
    """Corre en su propio hilo, para siempre, durante toda la vida del
    proceso -- carga el modelo una vez al principio y después atiende
    tareas de a una, en el mismo hilo que las cargó."""
    if not os.path.exists(MODEL_PATH):
        print(f"[AVISO] No encontré {MODEL_PATH} -- /predict queda deshabilitado. "
              "Entrenalo con train_landmarks_transformer.py o bajalo de Colab. "
              "El resto del server (dashboard, stats) funciona igual.")
    else:
        checkpoint = safe_torch_load(MODEL_PATH)
        class_names = checkpoint["class_names"]
        max_frames = checkpoint["max_frames"]
        input_dim = checkpoint.get("input_dim", 240)

        model = LipReadingConformer(num_classes=len(class_names), input_dim=input_dim).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print("Cargando detector de landmarks (HRNet/FAN)...")
        landmarker = build_landmarker()

        # Precalentamiento: la PRIMERA vez que HRNet corre, PyTorch intenta
        # compilar la red (torch.compile/dynamo), falla en Windows y cae a
        # modo eager -- ese intento fallido es lo que hace que la primera
        # predicción real tarde ~50-90s en vez de ~10-20s. Absorbemos ese
        # costo ACÁ, antes de aceptar requests, para que la primera
        # predicción de un usuario real ya sea rápida.
        if DEVICE == "cuda":
            try:
                print("Precalentando HRNet (primera compilación, puede tardar)...")
                dummy_frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(3)]
                detect_landmarks_batch(landmarker, dummy_frames)
                print("Precalentamiento listo.")
            except Exception as e:
                print(f"[AVISO] Precalentamiento falló ({e}), no es grave -- "
                      "la primera predicción real puede tardar más de lo normal.")

        _state.update(model=model, landmarker=landmarker, class_names=class_names, max_frames=max_frames)
        print(f"Listo. Device: {DEVICE}. Clases ({len(class_names)}): {class_names}")

    while True:
        func, args, result_future, loop = _gpu_queue.get()
        try:
            result = func(*args)
            loop.call_soon_threadsafe(result_future.set_result, result)
        except BaseException as e:  # noqa: BLE001 -- reportar CUALQUIER falla al que espera
            loop.call_soon_threadsafe(result_future.set_exception, e)


async def run_on_gpu_thread(func, *args):
    """Encola `func(*args)` para correr en el hilo dedicado de GPU y espera
    el resultado, sin bloquear el event loop de FastAPI mientras tanto
    (igual que hacía run_in_threadpool, pero siempre en el MISMO hilo)."""
    loop = asyncio.get_event_loop()
    result_future = loop.create_future()
    _gpu_queue.put((func, args, result_future, loop))
    return await result_future


@app.on_event("startup")
def load_everything():
    threading.Thread(target=_gpu_worker_loop, daemon=True, name="gpu-worker").start()


@app.on_event("shutdown")
def cleanup():
    landmarker = _state["landmarker"]
    if landmarker is not None and hasattr(landmarker, "close"):
        landmarker.close()


def preprocess_sequence(positions, max_frames):
    """positions: (T, N) ya continuo (sin huecos, ver fill_missing_frames).
    Suaviza y le agrega velocidad/aceleración, después padea/trunca a
    max_frames -- igual que en el entrenamiento y en probar_modelo.py."""
    smoothed = smooth_positions(positions)
    sequence_np = add_dynamics(smoothed)
    sequence = torch.from_numpy(sequence_np).float()

    T = sequence.shape[0]
    real_length = min(T, max_frames)
    if T < max_frames:
        padding = torch.zeros((max_frames - T,) + sequence.shape[1:], dtype=sequence.dtype)
        sequence = torch.cat([sequence, padding], dim=0)
    elif T > max_frames:
        sequence = sequence[:max_frames]

    mask = make_padding_mask(torch.tensor([real_length]), max_frames, sequence.device)
    return sequence.unsqueeze(0), mask  # (1, T, N), (1, T)


@app.get("/health")
def health():
    """La app la usa para chequear que el servidor está vivo y el modelo cargado."""
    return {"status": "ok", "device": DEVICE, "classes": _state["class_names"]}


# Extensiones de video reconocidas como "clip" -- .avi es lo que graba
# grabar_video_continuo.py (dataset), .webm/.mp4/.mov es lo que graba la
# cámara del navegador (pestaña "Grabar" de la app web) cuando /predict
# guarda una frase de riesgo real detectada en vivo (ver más abajo).
VIDEO_EXTENSIONS = (".avi", ".webm", ".mp4", ".mov")


def _count_files(folder, extensions=VIDEO_EXTENSIONS):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(extensions)])


@app.get("/dataset/stats")
def dataset_stats():
    """Progreso del dataset -- cuántos clips grabados y procesados hay por
    palabra, para mostrar el dashboard en la app."""
    words = set()
    if os.path.isdir(SESSIONS_DIR):
        words.update(d for d in os.listdir(SESSIONS_DIR)
                      if os.path.isdir(os.path.join(SESSIONS_DIR, d)))
    if os.path.isdir(DATASET_DIR):
        words.update(d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, d)))

    rows = []
    total_clips, total_processed = 0, 0
    for word in sorted(words):
        n_clips = _count_files(os.path.join(SESSIONS_DIR, word, "clips"))
        n_processed = _count_files(os.path.join(DATASET_DIR, word), ".pt")
        rows.append({"word": word, "clips": n_clips, "processed": n_processed})
        total_clips += n_clips
        total_processed += n_processed

    return {
        "goal_per_word": GOAL_PER_WORD,
        "total_words": len(rows),
        "total_clips": total_clips,
        "total_processed": total_processed,
        "words": rows,
    }


def _parse_clip_filename(word, filename):
    """De '<palabra>_<persona>_0001.avi' saca la persona, o None si el clip
    es viejo y no tiene persona en el nombre ('<palabra>_0001.avi')."""
    stem = os.path.splitext(filename)[0]
    prefix = word + "_"
    rest = stem[len(prefix):] if stem.lower().startswith(prefix.lower()) else stem
    if "_" in rest:
        person, _seq = rest.rsplit("_", 1)
        return person
    return None


@app.get("/dataset/recordings")
def dataset_recordings():
    """Sesiones de grabación agrupadas por fecha + frase + persona, para la
    pantalla de Capturas. Cada grupo incluye "thumbnail_file" -- el nombre
    del clip más reciente de ese grupo, para poder pedir una foto real del
    momento con GET /dataset/thumbnail (ver más abajo). Para los clips viejos
    del dataset de entrenamiento la app sigue mostrando el placeholder si
    la miniatura no carga -- acá no cambia nada de esa lógica."""
    if not os.path.isdir(SESSIONS_DIR):
        return {"recordings": []}

    grouped = defaultdict(list)
    for word in os.listdir(SESSIONS_DIR):
        clips_dir = os.path.join(SESSIONS_DIR, word, "clips")
        if not os.path.isdir(clips_dir):
            continue
        for filename in os.listdir(clips_dir):
            if not filename.lower().endswith(VIDEO_EXTENSIONS):
                continue
            path = os.path.join(clips_dir, filename)
            person = _parse_clip_filename(word, filename) or "desconocido"
            mtime = os.path.getmtime(path)
            date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            grouped[(date, word, person)].append((mtime, filename))

    recordings = []
    for (date, word, person), files in grouped.items():
        files.sort(key=lambda f: f[0], reverse=True)
        recordings.append({
            "date": date,
            # Hora de la captura más reciente del grupo -- no es una hora
            # por captura individual (acá se agrupa por día+palabra+persona,
            # no clip a clip), pero alcanza para mostrar "a qué hora fue la
            # última vez" en la app.
            "time": datetime.fromtimestamp(files[0][0]).strftime("%H:%M"),
            "word": word,
            "person": person,
            "count": len(files),
            "thumbnail_file": files[0][1],
        })
    recordings.sort(key=lambda r: r["date"], reverse=True)

    return {"recordings": recordings}


@app.get("/dataset/thumbnail")
def dataset_thumbnail(word: str, filename: str):
    """Devuelve un frame (JPEG) del medio de un clip guardado, para mostrar
    una foto real en el diálogo de captura de la app en vez del placeholder.
    `filename` se valida con basename -- nunca se deja salir de clips_dir."""
    safe_filename = os.path.basename(filename)
    safe_word = os.path.basename(word)
    path = os.path.join(SESSIONS_DIR, safe_word, "clips", safe_filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Clip no encontrado.")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=422, detail="No se pudo leer el clip.")

    # OJO: NO usar CAP_PROP_FRAME_COUNT + CAP_PROP_POS_FRAMES para "saltar"
    # al frame del medio -- los .webm que graba el navegador (MediaRecorder)
    # no traen ese índice bien armado y el seek falla en silencio. Leemos
    # todo secuencialmente (igual que hace /predict, que sabemos que sí
    # funciona con estos archivos) y nos quedamos con el del medio.
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise HTTPException(status_code=422, detail="No se pudo leer ningún frame del clip.")
    frame = frames[len(frames) // 2]

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la imagen.")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


def _save_web_capture(word, tmp_path, suffix):
    """Guarda una copia del clip grabado desde la app web (pestaña "Grabar")
    cuando /predict detecta una frase de riesgo real -- así queda registrada
    en Capturas/Frases (GET /dataset/recordings), igual que un clip grabado
    con grabar_video_continuo.py. Persona fija "web" para poder
    distinguirlas de las del dataset de entrenamiento. Nunca tira: si falla
    (ej. sin permisos de disco), la predicción igual se devuelve normal.

    Devuelve el nombre del archivo guardado (para que /predict lo mande de
    vuelta al cliente y la app pueda mostrar la foto real al toque, sin
    esperar a la próxima carga de Capturas), o None si falló el guardado.
    """
    try:
        clips_dir = os.path.join(SESSIONS_DIR, word, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        dest_name = f"{word}_web_{int(time.time() * 1000)}{suffix}"
        shutil.copy(tmp_path, os.path.join(clips_dir, dest_name))
        return dest_name
    except OSError as e:
        print(f"[AVISO] No se pudo guardar la captura web ({e}).")
        return None


# Ya NO hace falta un semáforo acá -- el hilo dedicado de GPU (ver
# _gpu_worker_loop) procesa una tarea a la vez desde su propia cola, así que
# la serialización ya está garantizada de por sí, sin nada extra.

# Tope duro por predicción individual -- ver uso en el endpoint /predict.
# Con el chunking de detect_landmarks_batch, un clip de 199 frames terminó
# bien en 67.4s. 90s deja margen de sobra arriba de eso.
PREDICT_HARD_TIMEOUT_S = 90


def _run_prediction_pipeline(tmp_path, suffix, landmarker, model, class_names, max_frames):
    """Todo el trabajo pesado (lectura de frames, HRNet, Conformer) -- código
    100% sincrónico y bloqueante a propósito, para correrlo en el hilo
    dedicado de GPU (ver _gpu_worker_loop) y no congelar el event loop de
    FastAPI mientras tarda (puede ser bastante, ver AVISO de detección en batch).

    Los print() con tiempos son a propósito -- si esto se cuelga nos dice
    EXACTAMENTE en qué etapa se quedó trabado (lectura de frames, detección
    HRNet, o inferencia del modelo), en vez de tener que adivinar."""
    t_start = time.time()
    if DEVICE == "cuda":
        # Limpieza de memoria ANTES de arrancar (no en cada sub-lote, eso ya
        # lo probamos y empeoraba el cuelgue) -- en uso intenso y seguido
        # (modo vigilancia, muchos /predict uno atrás del otro en el mismo
        # proceso) la memoria de la GPU se va fragmentando con cada request,
        # y eso parece ser lo que dispara el cuelgue del driver más adelante,
        # no el tamaño de un clip puntual.
        torch.cuda.empty_cache()
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="No se pudo leer el video enviado.")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # CAUSA REAL encontrada en producción: la cámara del navegador graba
        # en Full HD (1920x1080) -- 6.75x más píxeles por frame que los
        # 640x480 con los que se probó todo el pipeline (grabar_video_
        # continuo.py, probar_modelo.py). Ese tamaño es lo que realmente
        # disparaba el cuelgue, no la cantidad de frames. HRNet no necesita
        # esa resolución para encontrar la cara -- se achica ACÁ, antes de
        # meterla al pipeline, preservando el aspecto.
        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / w
            frame = cv2.resize(frame, (MAX_FRAME_WIDTH, round(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    print(f"[predict] frames leídos: {len(frames)} ({time.time() - t_start:.1f}s)")

    total = len(frames)
    if total == 0:
        # cv2 pudo abrir el archivo pero no decodificó ni un frame -- típico
        # de un códec que el build de OpenCV/ffmpeg del servidor no soporta
        # (ej. HEVC en .mov de algunos celulares), no de una toma corta.
        return {
            "valid": False,
            "reason": "no se pudo leer ningún frame del video -- el formato/códec "
                      "puede no ser compatible con el servidor (probá grabar en H.264/mp4)",
        }
    if total < MIN_FRAMES_VALID:
        return {
            "valid": False,
            "reason": f"toma muy corta ({total} frames, minimo {MIN_FRAMES_VALID}) "
                      "-- probablemente no se dijo la frase completa",
        }

    # Detección en BATCH: todo el clip al GPU de una sola vez en vez de
    # frame por frame -- ver detect_landmarks_batch en
    # extraer_landmarks_npy.py para el detalle de por qué es más rápido.
    landmarks_por_frame = detect_landmarks_batch(landmarker, frames)
    print(f"[predict] landmarks detectados ({time.time() - t_start:.1f}s)")

    positions = []
    detected = 0
    last_valid_vector = None
    for landmarks_px in landmarks_por_frame:
        vector = normalize_lip_landmarks(landmarks_px) if landmarks_px is not None else None
        if vector is not None and is_outlier(vector, last_valid_vector):
            vector = None  # glitch de detección, se rellena en vez de aceptarlo
        if vector is not None:
            detected += 1
            last_valid_vector = vector
        positions.append(vector)

    detection_rate = detected / total if total else 0
    filled = fill_missing_frames(positions)
    if filled is None or detection_rate < MIN_DETECTION_RATE_VALID:
        return {
            "valid": False,
            "reason": f"cara detectada solo en {detected}/{total} frames "
                      f"({detection_rate*100:.0f}%) -- encuadre malo, repetí la toma",
        }

    sequence, mask = preprocess_sequence(filled, max_frames)
    sequence, mask = sequence.to(DEVICE), mask.to(DEVICE)
    with torch.inference_mode():
        logits = model(sequence, src_key_padding_mask=mask)
        probs = torch.softmax(logits, dim=1)[0]
    print(f"[predict] modelo Conformer listo ({time.time() - t_start:.1f}s total)")

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    top_idx = int(sorted_idx[0].item())
    top_prob = float(sorted_probs[0].item())
    second_prob = float(sorted_probs[1].item()) if len(sorted_probs) > 1 else 0.0
    margin = top_prob - second_prob
    es_frase_de_riesgo = (
        class_names[top_idx] in RISK_WORDS
        and top_prob >= MIN_RISK_PROB
        and margin >= MIN_RISK_MARGIN
    )

    all_probs = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

    capture_file = None
    if es_frase_de_riesgo:
        capture_file = _save_web_capture(class_names[top_idx], tmp_path, suffix)

    return {
        "valid": True,
        "word": class_names[top_idx],
        "prob": top_prob,
        "detected": detected,
        "total": total,
        "es_frase_de_riesgo": es_frase_de_riesgo,
        # Nombre del clip guardado (o None) -- la app arma la URL de la
        # foto real con GET /dataset/thumbnail?word=...&filename=... usando
        # este valor, sin tener que esperar a recargar Capturas.
        "capture_file": capture_file,
        "all_probs": all_probs,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="El servidor todavía está cargando el modelo.")

    landmarker = _state["landmarker"]
    model = _state["model"]
    class_names = _state["class_names"]
    max_frames = _state["max_frames"]

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        try:
            # Tope duro -- si el hilo dedicado de GPU se cuelga de verdad,
            # esto corta la espera acá en vez de esperar para siempre. El
            # hilo colgado sigue vivo de fondo (Python no puede matarlo a la
            # fuerza), pero como es SIEMPRE el mismo hilo, ya no queda nada
            # más encolado detrás esperándolo salvo pedidos nuevos, que se
            # van a apilar hasta que el proceso se reinicie solo (ver abajo).
            return await asyncio.wait_for(
                run_on_gpu_thread(
                    _run_prediction_pipeline, tmp_path, suffix, landmarker, model, class_names, max_frames
                ),
                timeout=PREDICT_HARD_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            # El hilo colgado sigue vivo de fondo consumiendo VRAM -- en
            # Python no hay forma de matarlo a la fuerza. La única manera
            # real de recuperar esa memoria es que el PROCESO entero muera,
            # así que se autoreinicia solo (ver server/run.sh, que lo vuelve
            # a levantar apenas se cae). Se agenda la salida con un pequeño
            # delay para que esta respuesta 504 llegue a la app antes de que
            # el proceso se corte.
            print("[FATAL] /predict se colgó más de "
                  f"{PREDICT_HARD_TIMEOUT_S}s -- reiniciando el proceso para liberar la GPU.")
            asyncio.get_event_loop().call_later(2, lambda: os._exit(1))
            raise HTTPException(
                status_code=504,
                detail=f"La predicción tardó más de {PREDICT_HARD_TIMEOUT_S}s y se canceló -- "
                "el servidor se está reiniciando solo, probá de nuevo en unos segundos.",
            )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
