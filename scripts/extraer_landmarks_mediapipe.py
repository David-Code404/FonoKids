"""
extraer_landmarks_mediapipe.py
----------------------------------
Toma los clips individuales generados por recortar_video_palabras.py, corre
MediaPipe Face Landmarker (478 landmarks faciales) sobre cada frame, recorta
la boca usando los landmarks de labios (FACEMESH_LIPS), y guarda cada clip
como .pt + .pkl -- listo para train.py -- directo en dataset_pt/<palabra>/.

Intenta usar GPU (delegate GPU de MediaPipe Tasks); si no está disponible
en esta máquina/plataforma, cae a CPU automáticamente y te avisa.

Uso:
    python extraer_landmarks_mediapipe.py <palabra>
"""
import os
import sys
import json
import pickle
import urllib.request

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset_pt")
LABEL_MAP_PATH = os.path.join(DATASET_DIR, "label_map.json")

MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

ROI_SIZE = 112
MOUTH_MARGIN_RATIO = 1.3
MIN_DETECTION_CONFIDENCE = 0.7   # subido de 0.5: menos falsos positivos/detecciones inestables
SMOOTHING_ALPHA = 0.4  # suavizado exponencial del recuadro entre frames (0=no suaviza, 1=solo el ultimo frame)

# Índices de labios (interior+exterior) del esquema de 468 puntos de MediaPipe
# FaceMesh (constante estándar FACEMESH_LIPS -- hardcodeada porque esta versión
# de mediapipe (1.0.1, solo API "tasks") no expone el módulo "solutions" viejo).
LIP_INDICES = sorted({
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
})


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Descargando modelo de MediaPipe ({MODEL_URL})...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Guardado en: {MODEL_PATH}")


def build_landmarker():
    ensure_model_downloaded()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH,
                                          delegate=mp_python.BaseOptions.Delegate.GPU)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_face_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_DETECTION_CONFIDENCE,
    )
    try:
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        print(">>> MediaPipe corriendo en GPU. <<<")
        return landmarker
    except Exception as e:
        print("[AVISO] Esta build de MediaPipe para Windows no tiene soporte de GPU "
              "compilado (limitación del paquete, no del código) -- usando CPU.")
        print(f"       Detalle técnico: {e}")
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH,
                                              delegate=mp_python.BaseOptions.Delegate.CPU)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_DETECTION_CONFIDENCE,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        print(">>> MediaPipe corriendo en CPU. <<<")
        return landmarker


def landmarks_to_pixels(result, w, h):
    if not result.face_landmarks:
        return None
    face = result.face_landmarks[0]
    return np.array([[lm.x * w, lm.y * h] for lm in face], dtype=np.float32)


def compute_mouth_box_raw(landmarks_px, margin_ratio=MOUTH_MARGIN_RATIO):
    """Centro (cx,cy) y medio-lado del recuadro de boca, SIN suavizar ni
    recortar todavía -- valores en coordenadas de pixel flotantes."""
    lip_pts = landmarks_px[LIP_INDICES]
    center = lip_pts.mean(axis=0)
    half_side = max((lip_pts.max(axis=0) - lip_pts.min(axis=0)).max() / 2 * margin_ratio, 20)
    return center[0], center[1], half_side


def crop_from_box(frame, box, roi_size=ROI_SIZE):
    cx, cy, half_side = box
    h, w = frame.shape[:2]
    x1 = int(np.clip(cx - half_side, 0, w - 1))
    x2 = int(np.clip(cx + half_side, 0, w - 1))
    y1 = int(np.clip(cy - half_side, 0, h - 1))
    y2 = int(np.clip(cy + half_side, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)


def get_mouth_crop(frame, landmarks_px, roi_size=ROI_SIZE, margin_ratio=MOUTH_MARGIN_RATIO):
    """Sin suavizado temporal -- se mantiene para uso frame-a-frame suelto
    (ej. el preview en vivo de grabar_video_continuo.py)."""
    box = compute_mouth_box_raw(landmarks_px, margin_ratio)
    return crop_from_box(frame, box, roi_size)


def load_label_map():
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_label_map(label_map):
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)


def process_clip(landmarker, video_path, start_timestamp_ms):
    """start_timestamp_ms: MediaPipe en modo VIDEO exige timestamps siempre
    crecientes durante toda la vida del landmarker (no por clip) -- por eso
    se recibe y se devuelve el timestamp, en vez de reiniciarlo en 0 acá."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_ms = max(int(1000 / fps), 1)

    crops, landmarks_per_frame = [], []
    last_valid_crop = None
    smoothed_box = None  # (cx, cy, half_side) -- suavizado exponencial entre frames
    detected, total = 0, 0
    timestamp_ms = start_timestamp_ms

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += frame_ms

        landmarks_px = landmarks_to_pixels(result, frame.shape[1], frame.shape[0])

        crop = None
        if landmarks_px is not None:
            raw_box = compute_mouth_box_raw(landmarks_px)
            if smoothed_box is None:
                smoothed_box = raw_box
            else:
                a = SMOOTHING_ALPHA
                smoothed_box = tuple(
                    a * raw + (1 - a) * prev for raw, prev in zip(raw_box, smoothed_box)
                )
            crop = crop_from_box(frame, smoothed_box)

        landmarks_per_frame.append(landmarks_px.tolist() if landmarks_px is not None else None)

        if crop is not None:
            detected += 1
            last_valid_crop = crop
            crops.append(crop)
        elif last_valid_crop is not None:
            crops.append(last_valid_crop)

    cap.release()

    # Margen extra para que el próximo clip arranque siempre por encima del último usado
    next_timestamp_ms = timestamp_ms + frame_ms * 5

    if not crops:
        return None, None, detected, total, next_timestamp_ms

    sequence = np.stack(crops, axis=0)[:, :, :, ::-1].copy()  # BGR -> RGB
    sequence = torch.from_numpy(sequence).permute(0, 3, 1, 2).contiguous()  # (T,3,H,W) uint8
    return sequence, landmarks_per_frame, detected, total, next_timestamp_ms


def find_words_to_process():
    """Si se pasa una palabra por argumento, procesa solo esa. Si no, procesa
    TODAS las carpetas que haya dentro de sesiones_continuas/ automáticamente."""
    if len(sys.argv) >= 2:
        return [sys.argv[1].strip().lower()]

    if not os.path.exists(SESSIONS_DIR):
        return []
    return sorted(
        d for d in os.listdir(SESSIONS_DIR)
        if os.path.isdir(os.path.join(SESSIONS_DIR, d))
    )


def process_word(landmarker, label_map, word, timestamp_ms):
    clips_dir = os.path.join(SESSIONS_DIR, word, "clips")
    if not os.path.exists(clips_dir):
        print(f"[{word}] No encontré {clips_dir}, se omite.")
        return 0, 0, timestamp_ms

    clip_paths = sorted(
        os.path.join(clips_dir, f) for f in os.listdir(clips_dir) if f.endswith(".avi")
    )
    if not clip_paths:
        print(f"[{word}] No hay clips en {clips_dir}, se omite.")
        return 0, 0, timestamp_ms

    if word not in label_map:
        label_map[word] = len(label_map)
        save_label_map(label_map)
    label_index = label_map[word]
    print(f"\n[{word}] label {label_index}. {len(clip_paths)} clips a procesar.")

    out_dir = os.path.join(DATASET_DIR, word)
    os.makedirs(out_dir, exist_ok=True)

    ok_count, fail_count = 0, 0
    for i, clip_path in enumerate(clip_paths, 1):
        sequence, landmarks_per_frame, detected, total, timestamp_ms = process_clip(
            landmarker, clip_path, timestamp_ms
        )

        if sequence is None:
            print(f"  [{word} {i}/{len(clip_paths)}] {os.path.basename(clip_path)}: sin cara detectada, se omite.")
            fail_count += 1
            continue

        base_name = os.path.splitext(os.path.basename(clip_path))[0]
        pt_path = os.path.join(out_dir, f"{base_name}.pt")
        torch.save({"sequence": sequence, "label": label_index, "word": word}, pt_path)

        pkl_path = os.path.join(out_dir, f"{base_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"word": word, "label": label_index, "landmarks": landmarks_per_frame}, f)

        ok_count += 1
        if i % 25 == 0 or i == len(clip_paths):
            print(f"  [{word} {i}/{len(clip_paths)}] {base_name}: cara en {detected}/{total} frames -> guardado")

    print(f"[{word}] Listo. Guardados: {ok_count} | Omitidos (sin cara): {fail_count}")
    return ok_count, fail_count, timestamp_ms


def main():
    words = find_words_to_process()
    if not words:
        print(f"No hay ninguna palabra grabada en {SESSIONS_DIR}.")
        return

    print(f"Palabras a procesar ({len(words)}): {words}")
    landmarker = build_landmarker()
    label_map = load_label_map()

    total_ok, total_fail = 0, 0
    timestamp_ms = 0
    for word in words:
        ok, fail, timestamp_ms = process_word(landmarker, label_map, word, timestamp_ms)
        total_ok += ok
        total_fail += fail

    landmarker.close()
    print(f"\n=== TOTAL: {total_ok} guardados | {total_fail} omitidos (sin cara) ===")
    print(f"Dataset en: {DATASET_DIR}")
    print("Ya podés correr train.py apuntando a dataset_pt/.")


if __name__ == "__main__":
    main()
