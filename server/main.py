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
import os
import sys
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime

import cv2
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
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

# Igual que probar_modelo.py: el modelo SIEMPRE elige una de las clases
# entrenadas (no sabe decir "no sé"), así que esto es lo que distingue
# "sí es una frase de riesgo conocida" de "no es nada, no hacer caso".
MIN_RISK_PROB = 0.40
MIN_RISK_MARGIN = 0.15

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


@app.on_event("startup")
def load_everything():
    # El modelo es opcional -- si falta, el server igual arranca y sirve
    # /dataset/stats, /dataset/recordings, etc. Solo /predict queda
    # deshabilitado y avisa por qué, en vez de tirar abajo TODO el server.
    if not os.path.exists(MODEL_PATH):
        print(f"[AVISO] No encontré {MODEL_PATH} -- /predict queda deshabilitado. "
              "Entrenalo con train_landmarks_transformer.py o bajalo de Colab. "
              "El resto del server (dashboard, stats) funciona igual.")
        return

    checkpoint = safe_torch_load(MODEL_PATH)
    class_names = checkpoint["class_names"]
    max_frames = checkpoint["max_frames"]
    input_dim = checkpoint.get("input_dim", 240)

    model = LipReadingConformer(num_classes=len(class_names), input_dim=input_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Cargando detector de landmarks (HRNet/FAN)...")
    landmarker = build_landmarker()

    _state.update(model=model, landmarker=landmarker, class_names=class_names, max_frames=max_frames)
    print(f"Listo. Device: {DEVICE}. Clases ({len(class_names)}): {class_names}")


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


def _count_files(folder, extension):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(extension)])


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
        n_clips = _count_files(os.path.join(SESSIONS_DIR, word, "clips"), ".avi")
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
    stem = filename[:-4]  # sin ".avi"
    prefix = word + "_"
    rest = stem[len(prefix):] if stem.lower().startswith(prefix.lower()) else stem
    if "_" in rest:
        person, _seq = rest.rsplit("_", 1)
        return person
    return None


@app.get("/dataset/recordings")
def dataset_recordings():
    """Sesiones de grabación agrupadas por fecha + frase + persona, para la
    pantalla de Capturas (sin imágenes, solo fecha/frase/persona)."""
    if not os.path.isdir(SESSIONS_DIR):
        return {"recordings": []}

    grouped = defaultdict(int)
    for word in os.listdir(SESSIONS_DIR):
        clips_dir = os.path.join(SESSIONS_DIR, word, "clips")
        if not os.path.isdir(clips_dir):
            continue
        for filename in os.listdir(clips_dir):
            if not filename.lower().endswith(".avi"):
                continue
            path = os.path.join(clips_dir, filename)
            person = _parse_clip_filename(word, filename) or "desconocido"
            date = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
            grouped[(date, word, person)] += 1

    recordings = [
        {"date": date, "word": word, "person": person, "count": count}
        for (date, word, person), count in grouped.items()
    ]
    recordings.sort(key=lambda r: r["date"], reverse=True)

    return {"recordings": recordings}


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
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="No se pudo leer el video enviado.")

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        total = len(frames)
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
        with torch.no_grad():
            logits = model(sequence, src_key_padding_mask=mask)
            probs = torch.softmax(logits, dim=1)[0]

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        top_idx = int(sorted_idx[0].item())
        top_prob = float(sorted_probs[0].item())
        second_prob = float(sorted_probs[1].item()) if len(sorted_probs) > 1 else 0.0
        margin = top_prob - second_prob
        es_frase_de_riesgo = top_prob >= MIN_RISK_PROB and margin >= MIN_RISK_MARGIN

        all_probs = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

        return {
            "valid": True,
            "word": class_names[top_idx],
            "prob": top_prob,
            "es_frase_de_riesgo": es_frase_de_riesgo,
            "detected": detected,
            "total": total,
            "all_probs": all_probs,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
