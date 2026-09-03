"""
probar_modelo.py
-----------------
Prende la cámara, grabás una toma (C/S, igual que grabar_video_continuo.py)
y el modelo entrenado (mejor_modelo_landmarks_conformer.pth, bajado de
Colab o entrenado local con train_landmarks_transformer.py) te dice qué
frase cree que dijiste.

Usa MediaPipe Face Landmarker + la misma normalización de landmarks que
extraer_landmarks_npy.py, para que la predicción sea consistente con cómo
se entrenó (pipeline de landmarks puros + Conformer, sin recorte de imagen
ni ResNet).

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
import sys

import cv2
import torch

from train_landmarks_transformer import LipReadingConformer, make_padding_mask
from extraer_landmarks_npy import (
    build_landmarker,
    detect_frame_landmarks,
    normalize_lip_landmarks,
    fill_missing_frames,
    add_dynamics,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mejor_modelo_landmarks_conformer.pth")

MAX_CAMERAS_TO_CHECK = 5  # cuántos índices probar al buscar cámaras conectadas
FRAME_SIZE = (640, 480)
MIN_FRAMES_VALID = 15   # menos que esto = probablemente no dijiste la frase completa
MIN_DETECTION_RATE_VALID = 0.6  # menos del 60% de frames con cara detectada = toma mala
MAIN_WINDOW = "SpeakShadow - Probar modelo"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No encontré {MODEL_PATH}. Entrenalo con train_landmarks_transformer.py "
            "o bajalo de Colab (entrenar_landmarks_colab.ipynb) y ponelo en models/."
        )
    checkpoint = safe_torch_load(MODEL_PATH)
    class_names = checkpoint["class_names"]
    max_frames = checkpoint["max_frames"]
    input_dim = checkpoint.get("input_dim", 240)

    model = LipReadingConformer(num_classes=len(class_names), input_dim=input_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, max_frames


def preprocess_sequence(positions, max_frames):
    """positions: (T, 80) ya continuo (sin huecos, ver fill_missing_frames).
    Le agrega velocidad/aceleración -> (T, 240), y padea/trunca a max_frames,
    igual que en el entrenamiento."""
    sequence_np = add_dynamics(positions)  # (T, 240)
    sequence = torch.from_numpy(sequence_np).float()

    T = sequence.shape[0]
    real_length = min(T, max_frames)
    if T < max_frames:
        padding = torch.zeros((max_frames - T,) + sequence.shape[1:], dtype=sequence.dtype)
        sequence = torch.cat([sequence, padding], dim=0)
    elif T > max_frames:
        sequence = sequence[:max_frames]

    mask = make_padding_mask(torch.tensor([real_length]), max_frames, sequence.device)
    return sequence.unsqueeze(0), mask  # (1, T, 240), (1, T)


def predict_clip(model, landmarker, frames, class_names, max_frames, timestamp_ms):
    """Devuelve (resultado_dict_o_None, timestamp_ms_actualizado).
    resultado_dict tiene 'valid': False + 'reason' si la toma no vale
    (muy corta o cara mal detectada) -- en ese caso NO se corre el modelo.

    Igual que extraer_landmarks_npy.py: se guarda un valor por CADA frame
    (None si no se detectó cara) y se rellena con bfill/ffill, en vez de
    saltear frames -- así la predicción usa exactamente la misma
    continuidad temporal con la que se entrenó."""
    total = len(frames)

    if total < MIN_FRAMES_VALID:
        return {"valid": False, "reason": f"toma muy corta ({total} frames, "
                f"minimo {MIN_FRAMES_VALID}) -- probablemente no dijiste la frase completa"}, timestamp_ms

    positions = []
    detected = 0
    frame_ms = 40  # ~25 fps

    for frame in frames:
        landmarks_px, timestamp_ms = detect_frame_landmarks(landmarker, frame, timestamp_ms, frame_ms)
        vector = normalize_lip_landmarks(landmarks_px) if landmarks_px is not None else None

        if vector is not None:
            detected += 1
        positions.append(vector)

    detection_rate = detected / total
    filled = fill_missing_frames(positions)
    if filled is None or detection_rate < MIN_DETECTION_RATE_VALID:
        return {"valid": False, "reason": f"cara detectada solo en {detected}/{total} frames "
                f"({detection_rate*100:.0f}%) -- encuadre malo, repetí la toma"}, timestamp_ms

    sequence, mask = preprocess_sequence(filled, max_frames)
    sequence, mask = sequence.to(DEVICE), mask.to(DEVICE)
    with torch.no_grad():
        logits = model(sequence, src_key_padding_mask=mask)
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


def open_camera(index):
    """Abre una cámara con el backend correcto según el sistema operativo.

    En Windows, cv2.VideoCapture(index) SIN backend explícito a veces elige
    un backend equivocado (ej. "obsensor", pensado para cámaras de
    profundidad) que tira 'Camera index out of range' incluso con webcams
    normales conectadas -- por eso acá se fuerza DirectShow (CAP_DSHOW),
    que es el backend está­ndar y confiable para webcams en Windows."""
    if sys.platform == "win32":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def list_available_cameras(max_check=MAX_CAMERAS_TO_CHECK):
    """Prueba abrir varios índices de cámara y devuelve los que sí
    entregan imagen -- así detecta tanto la cámara integrada de la PC como
    cualquier webcam USB conectada, sin asumir cuál es el índice 0."""
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
    """Si hay una sola cámara, la usa directo. Si hay varias (ej. cámara
    integrada + webcam externa), te deja elegir cuál."""
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

    model, class_names, max_frames = load_model()
    print(f"Clases del modelo: {class_names}")
    print(f"Modelo cargado en {DEVICE}: {MODEL_PATH}")

    print("Cargando MediaPipe Face Landmarker...")
    landmarker = build_landmarker()

    cam_index = choose_camera()
    if cam_index is None:
        return

    cap = open_camera(cam_index)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara (índice {cam_index}).")
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

            result, timestamp_ms = predict_clip(model, landmarker, frames_buffer, class_names, max_frames, timestamp_ms)
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
