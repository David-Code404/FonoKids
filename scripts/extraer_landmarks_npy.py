"""
extraer_landmarks_npy.py
----------------------------
PARTE 1 del pipeline Transformer-puro (sin ResNet): recorre los clips
grabados (data/sesiones_continuas/<palabra>/clips/*.avi), corre MediaPipe
Face Landmarker, y por cada frame calcula un vector de 80 números:
    40 landmarks de labios (FACEMESH_LIPS) x 2 coordenadas (x, y)

Ese vector se guarda YA NORMALIZADO con invarianza a:
    - Traslación: se centra respecto al punto medio entre los ojos.
    - Rotación: se corrige el "roll" de la cabeza usando el ángulo entre
      los ojos, así da igual si la cabeza está un poco inclinada.
    - Escala: se divide por la distancia interocular, así da igual si
      estás más cerca o más lejos de la cámara.

Cada clip se guarda como un .npy con forma (T, 80), en carpetas separadas
por palabra -- así no hace falta cargar todo en RAM durante el entrenamiento.

Uso:
    python extraer_landmarks_npy.py            # procesa TODAS las palabras
    python extraer_landmarks_npy.py <palabra>   # procesa solo esa palabra

Lee de:    data/sesiones_continuas/<palabra>/clips/*.avi
Escribe en: data/landmarks_npy/<palabra>/*.npy
           data/landmarks_npy/label_map.json
"""
import os
import sys
import json
import math

import cv2
import numpy as np
import mediapipe as mp

from extraer_landmarks_mediapipe import build_landmarker, landmarks_to_pixels, LIP_INDICES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")
OUT_DIR = os.path.join(BASE_DIR, "data", "landmarks_npy")
LABEL_MAP_PATH = os.path.join(OUT_DIR, "label_map.json")

# Índices estándar de MediaPipe FaceMesh (478 puntos) para las esquinas
# externas de los ojos -- estables incluso mientras la boca se mueve mucho,
# por eso se usan como referencia para normalizar (no los landmarks de labios,
# que cambian de forma todo el tiempo con el habla).
RIGHT_EYE_OUTER = 33
LEFT_EYE_OUTER = 263


def load_label_map():
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_label_map(label_map):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)


def normalize_lip_landmarks(landmarks_px):
    """landmarks_px: array (478, 2) en pixeles. Devuelve un vector (80,)
    con los 40 puntos de labios normalizados (traslación+rotación+escala),
    o None si algo sale mal (ej. ojos muy pegados -> escala ~0)."""
    right_eye = landmarks_px[RIGHT_EYE_OUTER]
    left_eye = landmarks_px[LEFT_EYE_OUTER]

    eye_center = (right_eye + left_eye) / 2.0
    eye_vector = left_eye - right_eye
    scale = np.linalg.norm(eye_vector)
    if scale < 1e-3:
        return None
    angle = math.atan2(eye_vector[1], eye_vector[0])

    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

    lip_pts = landmarks_px[LIP_INDICES].astype(np.float64)          # (40, 2)
    translated = lip_pts - eye_center                                # traslación
    rotated = translated @ rotation.T                                # rotación
    normalized = rotated / scale                                     # escala

    return normalized.astype(np.float32).flatten()  # (80,)


def process_clip(landmarker, video_path, start_timestamp_ms):
    """Igual que en extraer_landmarks_mediapipe.py: el timestamp tiene que
    seguir subiendo durante toda la vida del landmarker, no reiniciarse
    por clip."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_ms = max(int(1000 / fps), 1)

    vectors = []
    last_valid_vector = None
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
        vector = normalize_lip_landmarks(landmarks_px) if landmarks_px is not None else None

        if vector is not None:
            detected += 1
            last_valid_vector = vector
            vectors.append(vector)
        elif last_valid_vector is not None:
            vectors.append(last_valid_vector)

    cap.release()
    next_timestamp_ms = timestamp_ms + frame_ms * 5

    if not vectors:
        return None, detected, total, next_timestamp_ms

    sequence = np.stack(vectors, axis=0)  # (T, 80) float32
    return sequence, detected, total, next_timestamp_ms


def find_words_to_process():
    if len(sys.argv) >= 2:
        return [sys.argv[1].strip().lower()]
    if not os.path.exists(SESSIONS_DIR):
        return []
    return sorted(d for d in os.listdir(SESSIONS_DIR)
                  if os.path.isdir(os.path.join(SESSIONS_DIR, d)))


def main():
    words = find_words_to_process()
    if not words:
        print(f"No hay ninguna palabra grabada en {SESSIONS_DIR}.")
        return

    print(f"Palabras a procesar ({len(words)}): {words}")
    landmarker = build_landmarker()
    label_map = load_label_map()

    timestamp_ms = 0
    total_ok, total_fail = 0, 0

    for word in words:
        clips_dir = os.path.join(SESSIONS_DIR, word, "clips")
        if not os.path.isdir(clips_dir):
            print(f"[{word}] no encontré {clips_dir}, se omite.")
            continue
        clip_paths = sorted(
            os.path.join(clips_dir, f) for f in os.listdir(clips_dir) if f.endswith(".avi")
        )
        if not clip_paths:
            print(f"[{word}] sin clips, se omite.")
            continue

        if word not in label_map:
            label_map[word] = len(label_map)
            save_label_map(label_map)

        out_dir = os.path.join(OUT_DIR, word)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{word}] label {label_map[word]}. {len(clip_paths)} clips a procesar.")
        for i, clip_path in enumerate(clip_paths, 1):
            sequence, detected, total, timestamp_ms = process_clip(landmarker, clip_path, timestamp_ms)

            if sequence is None:
                print(f"  [{word} {i}/{len(clip_paths)}] {os.path.basename(clip_path)}: sin cara detectada, se omite.")
                total_fail += 1
                continue

            base_name = os.path.splitext(os.path.basename(clip_path))[0]
            npy_path = os.path.join(out_dir, f"{base_name}.npy")
            np.save(npy_path, sequence)

            total_ok += 1
            if i % 25 == 0 or i == len(clip_paths):
                print(f"  [{word} {i}/{len(clip_paths)}] {base_name}: cara en {detected}/{total} frames -> "
                      f"guardado {sequence.shape}")

    landmarker.close()
    print(f"\n=== TOTAL: {total_ok} guardados | {total_fail} omitidos (sin cara) ===")
    print(f"Dataset en: {OUT_DIR}")


if __name__ == "__main__":
    main()
