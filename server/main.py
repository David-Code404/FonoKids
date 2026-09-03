"""
server/main.py
---------------
API HTTP para la app de Flutter. Recibe un video corto (grabado desde el
celular), corre EXACTAMENTE el mismo pipeline que probar_modelo.py
(MediaPipe Face Landmarker -> recorte de boca -> VisualSpeechTransformer)
y devuelve la palabra predicha en JSON.

Uso:
    python server/main.py
    (por defecto escucha en 0.0.0.0:8000 -- accesible desde el celular
    si está en la misma red WiFi que esta PC, usando la IP local de la PC)
"""
import os
import sys
import json
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from train import VisualSpeechTransformer, safe_torch_load  # noqa: E402
from extraer_landmarks_npy import (  # noqa: E402
    build_landmarker,
    landmarks_to_pixels,
    get_mouth_crop,
)

DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset_pt")
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")
LABEL_MAP_PATH = os.path.join(DATASET_DIR, "label_map.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mejor_modelo_speakshadow.pt")
GOAL_PER_WORD = 500

MAX_FRAMES = 25
MIN_FRAMES_VALID = 15          # menos que esto = probablemente no se dijo la frase completa
MIN_DETECTION_RATE_VALID = 0.6  # menos del 60% de frames con cara detectada = toma mala
FRAME_MS = 40  # ~25 fps, para los timestamps que exige MediaPipe en modo VIDEO

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="SpeakShadow API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global: modelo y landmarker se cargan UNA sola vez al arrancar,
# no en cada request (cargar el modelo por request sería lentísimo).
_state = {"model": None, "landmarker": None, "class_names": None, "timestamp_ms": 0}


def load_class_names():
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"No encontré {LABEL_MAP_PATH}.")
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = max(label_map.values()) + 1
    names = ["?"] * num_classes
    for word, idx in label_map.items():
        names[idx] = word
    return names


@app.on_event("startup")
def load_everything():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"No encontré {MODEL_PATH}. Entrená el modelo (scripts/train.py) o bajalo "
            "de Colab y ponelo en la carpeta models/ antes de levantar el servidor."
        )
    class_names = load_class_names()
    model = VisualSpeechTransformer(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(safe_torch_load(MODEL_PATH))
    model.eval()

    print("Cargando MediaPipe Face Landmarker...")
    landmarker = build_landmarker()

    _state.update(model=model, landmarker=landmarker, class_names=class_names)
    print(f"Listo. Device: {DEVICE}. Clases ({len(class_names)}): {class_names}")


@app.on_event("shutdown")
def cleanup():
    if _state["landmarker"] is not None:
        _state["landmarker"].close()


def preprocess_sequence(crops):
    sequence = np.stack(crops, axis=0)[:, :, :, ::-1].copy()  # BGR -> RGB
    sequence = torch.from_numpy(sequence).permute(0, 3, 1, 2).contiguous().float()
    sequence = sequence / 255.0
    sequence = (sequence - IMAGENET_MEAN) / IMAGENET_STD

    T = sequence.shape[0]
    if T < MAX_FRAMES:
        padding = torch.zeros((MAX_FRAMES - T,) + sequence.shape[1:], dtype=sequence.dtype)
        sequence = torch.cat([sequence, padding], dim=0)
    elif T > MAX_FRAMES:
        sequence = sequence[:MAX_FRAMES]

    return sequence.unsqueeze(0)  # (1, T, 3, H, W)


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

        crops = []
        last_valid_crop = None
        detected = 0
        timestamp_ms = _state["timestamp_ms"]

        import mediapipe as mp  # import local: evita cargarlo si el modelo no está listo

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += FRAME_MS

            landmarks_px = landmarks_to_pixels(result, frame.shape[1], frame.shape[0])
            crop = get_mouth_crop(frame, landmarks_px) if landmarks_px is not None else None

            if crop is not None:
                detected += 1
                last_valid_crop = crop
                crops.append(crop)
            elif last_valid_crop is not None:
                crops.append(last_valid_crop)

        # Margen extra para que el próximo request arranque siempre con timestamp mayor
        # (MediaPipe en modo VIDEO exige timestamps siempre crecientes durante toda la
        # vida del landmarker, no solo dentro de un video).
        _state["timestamp_ms"] = timestamp_ms + FRAME_MS * 5

        detection_rate = detected / total if total else 0
        if not crops or detection_rate < MIN_DETECTION_RATE_VALID:
            return {
                "valid": False,
                "reason": f"cara detectada solo en {detected}/{total} frames "
                          f"({detection_rate*100:.0f}%) -- encuadre malo, repetí la toma",
            }

        sequence = preprocess_sequence(crops).to(DEVICE)
        with torch.no_grad():
            logits = model(sequence)
            probs = torch.softmax(logits, dim=1)[0]

        top_idx = int(torch.argmax(probs).item())
        top_prob = float(probs[top_idx].item())
        all_probs = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

        return {
            "valid": True,
            "word": class_names[top_idx],
            "prob": top_prob,
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
