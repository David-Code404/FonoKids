"""
probar_modelo.py
-----------------
Prende la cámara, grabás una toma (C/S, igual que grabar_video_continuo.py)
y el modelo entrenado (mejor_modelo_speakshadow.pt, bajado de Colab) te dice
qué palabra cree que dijiste.

Usa MediaPipe Face Landmarker para el recorte de boca -- el MISMO método que
extraer_landmarks_mediapipe.py, para que la predicción sea consistente con
cómo se entrenó.

IMPORTANTE: para saber si el modelo realmente aprendió a leer labios (y no
memorizó detalles de la sesión de grabación), probalo con tomas grabadas
AHORA, en otro momento distinto al que usaste para entrenar.

Controles:
    C -> Empieza a grabar
    S -> Detiene y predice
    Q -> Sale

Uso:
    python probar_modelo.py
"""
import os
import json

import cv2
import numpy as np
import torch

from train import VisualSpeechTransformer, safe_torch_load
from extraer_landmarks_mediapipe import build_landmarker, landmarks_to_pixels, get_mouth_crop

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset_pt")
LABEL_MAP_PATH = os.path.join(DATASET_DIR, "label_map.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mejor_modelo_speakshadow.pt")

CAM_INDEX = 0
FRAME_SIZE = (640, 480)
MAX_FRAMES = 25
MIN_FRAMES_VALID = 15   # menos que esto = probablemente no dijiste la frase completa
MIN_DETECTION_RATE_VALID = 0.6  # menos del 60% de frames con cara detectada = toma mala
MAIN_WINDOW = "SpeakShadow - Probar modelo"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def load_model(num_classes):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No encontré {MODEL_PATH}. Bajalo de Colab (/content/mejor_modelo_speakshadow.pt) "
            "y ponelo en la carpeta models/."
        )
    model = VisualSpeechTransformer(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(safe_torch_load(MODEL_PATH))
    model.eval()
    return model


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


def predict_clip(model, landmarker, frames, class_names, timestamp_ms):
    """Devuelve (resultado_dict_o_None, timestamp_ms_actualizado).
    resultado_dict tiene 'valid': False + 'reason' si la toma no vale
    (muy corta o cara mal detectada) -- en ese caso NO se corre el modelo."""
    total = len(frames)

    if total < MIN_FRAMES_VALID:
        return {"valid": False, "reason": f"toma muy corta ({total} frames, "
                f"minimo {MIN_FRAMES_VALID}) -- probablemente no dijiste la frase completa"}, timestamp_ms

    crops = []
    last_valid_crop = None
    detected = 0
    frame_ms = 40  # ~25 fps

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += frame_ms

        landmarks_px = landmarks_to_pixels(result, frame.shape[1], frame.shape[0])
        crop = get_mouth_crop(frame, landmarks_px) if landmarks_px is not None else None

        if crop is not None:
            detected += 1
            last_valid_crop = crop
            crops.append(crop)
        elif last_valid_crop is not None:
            crops.append(last_valid_crop)

    detection_rate = detected / total
    if not crops or detection_rate < MIN_DETECTION_RATE_VALID:
        return {"valid": False, "reason": f"cara detectada solo en {detected}/{total} frames "
                f"({detection_rate*100:.0f}%) -- encuadre malo, repetí la toma"}, timestamp_ms

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
    }, timestamp_ms


def draw_overlay(frame, recording, last_result):
    h, w = frame.shape[:2]
    status_text, color = ("GRABANDO", (0, 0, 255)) if recording else ("EN PAUSA", (0, 200, 0))
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if recording:
        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)

    if last_result is not None:
        word, prob = last_result
        safe_word = word.encode("ascii", "replace").decode("ascii")
        cv2.putText(frame, f"Prediccion: {safe_word} ({prob*100:.1f}%)", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, "C: grabar | S: detener y predecir | Q: salir",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def main():
    if DEVICE != "cuda":
        print("[AVISO] No se detectó GPU (CUDA) -- corriendo en CPU, va a ser mucho más lento.")
    else:
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    class_names = load_class_names()
    print(f"Clases del modelo: {class_names}")
    model = load_model(num_classes=len(class_names))
    print(f"Modelo cargado en {DEVICE}: {MODEL_PATH}")

    print("Cargando MediaPipe Face Landmarker...")
    landmarker = build_landmarker()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

    recording, frames_buffer, last_result = False, [], None
    timestamp_ms = 0

    print("Cámara iniciada. C=grabar, S=detener y predecir, Q=salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if recording:
            frames_buffer.append(frame.copy())

        cv2.imshow(MAIN_WINDOW, draw_overlay(frame.copy(), recording, last_result))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and not recording:
            recording, frames_buffer, last_result = True, [], None
            print("-> Grabando toma de prueba...")

        elif key == ord('s') and recording:
            recording = False
            print(f"-> Prediciendo sobre {len(frames_buffer)} frames...")

            result, timestamp_ms = predict_clip(model, landmarker, frames_buffer, class_names, timestamp_ms)
            frames_buffer = []

            if not result["valid"]:
                print(f"  [TOMA NO VÁLIDA] {result['reason']}")
                last_result = None
                continue

            last_result = (result["word"], result["prob"])
            print(f"  Cara detectada en {result['detected']}/{result['total']} frames")
            print(f"  -> Predicción: '{result['word']}' con {result['prob']*100:.1f}% de confianza")
            print(f"  Probabilidades: { {k: round(v,3) for k,v in result['all_probs'].items()} }")

        elif key == ord('q'):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
