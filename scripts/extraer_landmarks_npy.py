import os
import sys
import json
import math
import urllib.request

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUTA_CLIPS = os.path.join(BASE_DIR, "data", "sesiones_continuas")

DETECTOR_BACKEND = "hrnet"

HRNET_BATCH_CHUNK_SIZE = 4

OUT_DIR = os.path.join(BASE_DIR, "data", "landmarks_npy")
LABEL_MAP_PATH = os.path.join(OUT_DIR, "label_map.json")

MEDIAPIPE_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

MIN_DETECTION_CONFIDENCE = 0.7
ROI_SIZE = 112
MOUTH_MARGIN_RATIO = 1.3

MIN_DETECTION_RATE_TRAIN = 0.6

MAX_CAMERAS_TO_CHECK = 5

SCALE_EPSILON = 1e-6

if DETECTOR_BACKEND == "hrnet":
    LIP_INDICES = list(range(48, 68))
    MOUTH_RIGHT_CORNER = 48
    MOUTH_LEFT_CORNER = 54
else:
    LIP_INDICES = sorted({
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        185, 40, 39, 37, 0, 267, 269, 270, 409,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        191, 80, 81, 82, 13, 312, 311, 310, 415,
    })
    MOUTH_RIGHT_CORNER = 61
    MOUTH_LEFT_CORNER = 291

POSITION_DIM = len(LIP_INDICES) * 2
LANDMARK_DIM = POSITION_DIM * 3


def build_landmarker():
    if DETECTOR_BACKEND == "hrnet":
        return _build_hrnet_detector()
    return _build_mediapipe_landmarker()


def detect_landmarks_batch(landmarker, frames_bgr, timestamp_ms=0, frame_ms=40):
    if DETECTOR_BACKEND != "hrnet" or not frames_bgr:
        resultados = []
        for frame in frames_bgr:
            landmarks_px, timestamp_ms = detect_frame_landmarks(landmarker, frame, timestamp_ms, frame_ms)
            resultados.append(landmarks_px)
        return resultados

    import torch

    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

    device = landmarker.face_detector.device
    chunk_size = HRNET_BATCH_CHUNK_SIZE
    detected_faces = []
    try:
        for start in range(0, len(frames_rgb), chunk_size):
            chunk = frames_rgb[start:start + chunk_size]
            batch = torch.stack([
                torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1) for f in chunk
            ]).to(device)
            try:
                chunk_faces = landmarker.face_detector.detect_from_batch(batch)
            except torch.cuda.OutOfMemoryError:
                del batch
                print(f"[AVISO] Sub-lote de {len(chunk)} frames sin memoria, "
                      "reintentando de a un frame.")
                chunk_faces = []
                for frame in chunk:
                    single = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0).to(device)
                    chunk_faces.extend(landmarker.face_detector.detect_from_batch(single))
                    del single
            detected_faces.extend(chunk_faces)
            del batch
    except Exception as e:
        print(f"[AVISO] Deteccion en batch fallo ({e}), usando frame por frame.")
        detected_faces = None

    resultados = []
    for i, frame_rgb in enumerate(frames_rgb):
        if detected_faces is None:
            preds = landmarker.get_landmarks_from_image(frame_rgb)
        else:
            faces = detected_faces[i] if i < len(detected_faces) else []
            if not len(faces):
                resultados.append(None)
                continue
            preds = landmarker.get_landmarks_from_image(frame_rgb, detected_faces=faces)
        resultados.append(preds[0].astype(np.float32) if preds else None)
    return resultados


def detect_frame_landmarks(landmarker, frame_bgr, timestamp_ms, frame_ms=40):
    if DETECTOR_BACKEND == "hrnet":
        return _detect_hrnet(landmarker, frame_bgr), timestamp_ms
    return _detect_mediapipe(landmarker, frame_bgr, timestamp_ms, frame_ms)


def _ensure_mediapipe_model_downloaded():
    if os.path.exists(MEDIAPIPE_MODEL_PATH):
        return
    print(f"Descargando modelo de MediaPipe ({MEDIAPIPE_MODEL_URL})...")
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
    print(f"Guardado en: {MEDIAPIPE_MODEL_PATH}")


def _build_mediapipe_landmarker():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    _ensure_mediapipe_model_downloaded()
    base_options = mp_python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH,
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
        base_options = mp_python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH,
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


def _detect_mediapipe(landmarker, frame_bgr, timestamp_ms, frame_ms):
    import mediapipe as mp
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    landmarks_px = landmarks_to_pixels(result, frame_bgr.shape[1], frame_bgr.shape[0])
    return landmarks_px, timestamp_ms + frame_ms


def _build_hrnet_detector():
    import face_alignment
    import torch
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[AVISO] No hay GPU disponible -- HRNet/FAN en CPU va a ser MUY lento "
              "(mucho más que MediaPipe en CPU). Se recomienda GPU para este backend.")

    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=device, flip_input=False, compile=False,
    )
    print(f">>> Detector HRNet/FAN corriendo en {device.upper()}. <<<")
    return detector


def _detect_hrnet(landmarker, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preds = landmarker.get_landmarks(rgb)
    if not preds:
        return None
    return preds[0].astype(np.float32)


def compute_mouth_box_raw(landmarks_px, margin_ratio=MOUTH_MARGIN_RATIO):
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
    box = compute_mouth_box_raw(landmarks_px, margin_ratio)
    return crop_from_box(frame, box, roi_size)


def normalize_lip_landmarks(landmarks_px):
    right_corner = landmarks_px[MOUTH_RIGHT_CORNER]
    left_corner = landmarks_px[MOUTH_LEFT_CORNER]

    mouth_center = (right_corner + left_corner) / 2.0
    mouth_vector = left_corner - right_corner
    scale = np.linalg.norm(mouth_vector)
    if scale < 1e-3:
        return None
    angle = math.atan2(mouth_vector[1], mouth_vector[0])

    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

    lip_pts = landmarks_px[LIP_INDICES].astype(np.float64)
    translated = lip_pts - mouth_center
    rotated = translated @ rotation.T
    normalized = rotated / (scale + SCALE_EPSILON)

    return normalized.astype(np.float32).flatten()


OUTLIER_MAX_JUMP = 0.15


def is_outlier(vector, last_valid_vector):
    if last_valid_vector is None:
        return False
    return float(np.abs(vector - last_valid_vector).max()) > OUTLIER_MAX_JUMP


def fill_missing_frames(positions):
    primeros_validos = [i for i, p in enumerate(positions) if p is not None]
    if not primeros_validos:
        return None

    primer_valido = primeros_validos[0]
    ultimo_valido_idx = primeros_validos[-1]
    relleno = list(positions)

    for i in range(primer_valido):
        relleno[i] = relleno[primer_valido]

    i = primer_valido
    while i < ultimo_valido_idx:
        if relleno[i + 1] is None:
            j = i + 1
            while relleno[j] is None:
                j += 1
            inicio, fin = relleno[i], relleno[j]
            pasos = j - i
            for k in range(1, pasos):
                t = k / pasos
                relleno[i + k] = (1 - t) * inicio + t * fin
            i = j
        else:
            i += 1

    for i in range(ultimo_valido_idx + 1, len(relleno)):
        relleno[i] = relleno[ultimo_valido_idx]

    return np.stack(relleno, axis=0).astype(np.float32)


SMOOTHING_ALPHA = 0.4


def smooth_positions(positions, alpha=SMOOTHING_ALPHA):
    smoothed = np.empty_like(positions)
    smoothed[0] = positions[0]
    for t in range(1, positions.shape[0]):
        smoothed[t] = alpha * positions[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed.astype(np.float32)


def add_dynamics(positions):
    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]

    acceleration = np.zeros_like(positions)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    return np.concatenate([positions, velocity, acceleration], axis=1).astype(np.float32)


def process_clip(landmarker, video_path, start_timestamp_ms):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_ms = max(int(1000 / fps), 1)

    positions = []
    detected, total = 0, 0
    last_valid_vector = None
    timestamp_ms = start_timestamp_ms

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1

        landmarks_px, timestamp_ms = detect_frame_landmarks(landmarker, frame, timestamp_ms, frame_ms)
        vector = normalize_lip_landmarks(landmarks_px) if landmarks_px is not None else None

        if vector is not None and is_outlier(vector, last_valid_vector):
            vector = None

        if vector is not None:
            detected += 1
            last_valid_vector = vector
        positions.append(vector)

    cap.release()
    next_timestamp_ms = timestamp_ms + frame_ms * 5

    filled = fill_missing_frames(positions)
    if filled is None:
        return None, detected, total, next_timestamp_ms

    smoothed = smooth_positions(filled)
    sequence = add_dynamics(smoothed)
    return sequence, detected, total, next_timestamp_ms


def open_camera(index):
    if sys.platform == "win32":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def list_available_cameras(max_check=MAX_CAMERAS_TO_CHECK):
    found = []
    for i in range(max_check):
        cap = open_camera(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found


def choose_camera():
    print("Buscando cámaras disponibles...")
    cameras = list_available_cameras()
    if not cameras:
        print("No encontré ninguna cámara conectada.")
        return None

    if len(cameras) == 1:
        print(f"Encontré 1 cámara (índice {cameras[0]}), la uso.")
        return cameras[0]

    print(f"Encontré {len(cameras)} cámaras: {cameras}")
    print("(el índice 0 suele ser la cámara integrada de la PC/notebook; "
          "los demás índices suelen ser webcams USB externas)")
    while True:
        choice = input(f"¿Cuál querés usar? (número de índice, ej. {cameras[0]}): ").strip()
        try:
            idx = int(choice)
            if idx in cameras:
                return idx
        except ValueError:
            pass
        print(f"Opción inválida -- elegí uno de {cameras}.")


def find_words_to_process():
    if len(sys.argv) >= 2:
        return [sys.argv[1].strip().lower()]
    if not os.path.exists(RUTA_CLIPS):
        return []
    return sorted(d for d in os.listdir(RUTA_CLIPS)
                  if os.path.isdir(os.path.join(RUTA_CLIPS, d)))


def main():
    words = find_words_to_process()
    if not words:
        print(f"No hay ninguna palabra grabada en {RUTA_CLIPS}.")
        return

    print(f"Palabras a procesar ({len(words)}): {words}")
    print(f"Dimensión de salida por frame: {LANDMARK_DIM} "
          f"(posición {POSITION_DIM} + velocidad {POSITION_DIM} + aceleración {POSITION_DIM})")
    landmarker = build_landmarker()
    label_map = {}
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)

    timestamp_ms = 0
    total_ok, total_fail = 0, 0

    for word in words:
        clips_dir = os.path.join(RUTA_CLIPS, word, "clips")
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
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
                json.dump(label_map, f, ensure_ascii=False, indent=2)

        out_dir = os.path.join(OUT_DIR, word)
        os.makedirs(out_dir, exist_ok=True)

        pendientes = [
            cp for cp in clip_paths
            if not os.path.exists(os.path.join(out_dir, os.path.splitext(os.path.basename(cp))[0] + ".npy"))
        ]
        ya_hechos = len(clip_paths) - len(pendientes)
        if ya_hechos:
            print(f"\n[{word}] {ya_hechos} clips ya tenían landmarks (se saltean), "
                  f"{len(pendientes)} nuevos a procesar.")
        else:
            print(f"\n[{word}] label {label_map[word]}. {len(pendientes)} clips a procesar.")

        for i, clip_path in enumerate(pendientes, 1):
            sequence, detected, total, timestamp_ms = process_clip(landmarker, clip_path, timestamp_ms)

            if sequence is None:
                print(f"  [{word} {i}/{len(pendientes)}] {os.path.basename(clip_path)}: "
                      f"NINGÚN frame detectó cara ({total} frames) -- clip descartado.")
                total_fail += 1
                continue

            detection_rate = detected / total if total else 0
            if detection_rate < MIN_DETECTION_RATE_TRAIN:
                print(f"  [{word} {i}/{len(pendientes)}] {os.path.basename(clip_path)}: "
                      f"cara en {detected}/{total} frames ({detection_rate*100:.0f}%) -- "
                      "por debajo del ideal, pero se guarda igual (rellenado bfill/ffill).")

            base_name = os.path.splitext(os.path.basename(clip_path))[0]
            npy_path = os.path.join(out_dir, f"{base_name}.npy")
            np.save(npy_path, sequence)

            total_ok += 1
            if i % 25 == 0 or i == len(pendientes):
                print(f"  [{word} {i}/{len(pendientes)}] {base_name}: cara en {detected}/{total} frames "
                      f"(resto rellenado bfill/ffill) -> guardado {sequence.shape}")

    if hasattr(landmarker, "close"):
        landmarker.close()
    print(f"\n=== TOTAL: {total_ok} guardados | {total_fail} descartados (sin cara en ningún frame) ===")
    print(f"Dataset en: {OUT_DIR}")


if __name__ == "__main__":
    main()
