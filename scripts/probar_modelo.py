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
from datetime import datetime

import cv2
import torch

from train_landmarks_transformer import LipReadingConformer, make_padding_mask
from extraer_landmarks_npy import (
    build_landmarker,
    detect_frame_landmarks,
    detect_landmarks_batch,
    normalize_lip_landmarks,
    fill_missing_frames,
    smooth_positions,
    add_dynamics,
    is_outlier,
    open_camera,
    list_available_cameras,
    choose_camera,
    _build_mediapipe_landmarker,
    _detect_mediapipe,
)

# Puntos de referencia SOLO para dibujar el overlay en vivo (esquema de 478
# puntos de MediaPipe) -- independiente de qué DETECTOR_BACKEND se use para
# la extracción/predicción real. Se usa MediaPipe acá porque es rápido
# (HRNet a ~3fps trabaría la vista en vivo, igual que en grabar_video_
# continuo.py).
PREVIEW_LIP_INDICES = sorted({
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
})
PREVIEW_RIGHT_EYE_OUTER, PREVIEW_LEFT_EYE_OUTER = 33, 263
LIVE_DETECT_EVERY = 2  # correr el detector del preview cada N frames

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mejor_modelo_landmarks_conformer.pth")
CAPTURAS_DIR = os.path.join(BASE_DIR, "capturas")

FRAME_SIZE = (640, 480)
MIN_FRAMES_VALID = 15   # menos que esto = probablemente no dijiste la frase completa
MIN_DETECTION_RATE_VALID = 0.6  # menos del 60% de frames con cara detectada = toma mala
MAIN_WINDOW = "SpeakShadow - Probar modelo"

# El modelo SIEMPRE tiene que elegir una de las 57 frases (es un clasificador
# de conjunto cerrado, no sabe decir "no sé") -- así que si decís algo que no
# es ninguna frase de riesgo (ej. "hola", "adiós"), igual te va a devolver
# la frase que más se le pareció, con confianza baja. Este umbral es lo que
# distingue "SÍ es una frase conocida" de "no es nada, no hacer caso":
# si no lo supera, se trata como que NO se dijo ninguna frase de riesgo.
MIN_RISK_PROB = 0.40      # la clase top tiene que superar esto
MIN_RISK_MARGIN = 0.15    # y sacarle esta diferencia mínima a la 2da opción

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
    Suaviza (igual que el extractor) y le agrega velocidad/aceleración ->
    (T, 240), y padea/trunca a max_frames, igual que en el entrenamiento."""
    smoothed = smooth_positions(positions)
    sequence_np = add_dynamics(smoothed)  # (T, 240)
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
    last_valid_vector = None
    frame_ms = 40  # ~25 fps

    # Detección en BATCH: manda el clip entero al GPU de una sola vez en vez
    # de detectar cara por cara en cada frame -- esto es lo que hace que la
    # predicción salga en segundos y no en 15-20s con HRNet.
    landmarks_por_frame = detect_landmarks_batch(landmarker, frames, timestamp_ms, frame_ms)
    if landmarks_por_frame:
        timestamp_ms += frame_ms * len(frames)

    for landmarks_px in landmarks_por_frame:
        vector = normalize_lip_landmarks(landmarks_px) if landmarks_px is not None else None

        if vector is not None and is_outlier(vector, last_valid_vector):
            vector = None  # glitch de detección, se rellena en vez de aceptarlo

        if vector is not None:
            detected += 1
            last_valid_vector = vector
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
    }, timestamp_ms


def draw_landmarks_points(frame, landmarks_px):
    """Dibuja los puntos de MediaPipe sobre el frame, igual que la imagen de
    referencia: rojo = resto de la malla facial (no se usa), amarillo =
    esquinas de los ojos (referencia de normalización), verde = los 40
    puntos de labios (los únicos que ve el modelo)."""
    if landmarks_px is None:
        return frame
    for i, (x, y) in enumerate(landmarks_px):
        if i in PREVIEW_LIP_INDICES:
            color, radius = (0, 255, 0), 3
        elif i in (PREVIEW_RIGHT_EYE_OUTER, PREVIEW_LEFT_EYE_OUTER):
            color, radius = (0, 255, 255), 4
        else:
            color, radius = (0, 0, 255), 1
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    return frame


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


def save_capture(frame_bgr, landmarks_px, word, prob):
    """Guarda una foto (.jpg) del frame con un recuadro alrededor de la cara
    y la frase detectada escrita arriba -- una captura de "quién dijo qué"
    para cada predicción, en CAPTURAS_DIR."""
    os.makedirs(CAPTURAS_DIR, exist_ok=True)
    frame_out = frame_bgr.copy()
    h, w = frame_out.shape[:2]

    if landmarks_px is not None:
        x1, y1 = landmarks_px[:, 0].min(), landmarks_px[:, 1].min()
        x2, y2 = landmarks_px[:, 0].max(), landmarks_px[:, 1].max()
        margin = int(0.25 * max(x2 - x1, y2 - y1))
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(w - 1, int(x2) + margin)
        y2 = min(h - 1, int(y2) + margin)
        cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_y = max(25, y1 - 12)
    else:
        label_y = 30

    texto = f"{word.replace('_', ' ')} ({prob*100:.0f}%)"
    safe_texto = texto.encode("ascii", "replace").decode("ascii")
    cv2.putText(frame_out, safe_texto, (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(CAPTURAS_DIR, f"{word}_{timestamp}.jpg")
    cv2.imwrite(out_path, frame_out)
    return out_path


def main():
    if DEVICE != "cuda":
        print("[AVISO] No se detectó GPU (CUDA) -- corriendo en CPU, va a ser mucho más lento.")
    else:
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    model, class_names, max_frames = load_model()
    print(f"Clases del modelo: {class_names}")
    print(f"Modelo cargado en {DEVICE}: {MODEL_PATH}")

    print("Cargando detector de landmarks...")
    landmarker = build_landmarker()

    print("Cargando MediaPipe para el preview en vivo (rápido, solo visual)...")
    preview_landmarker = _build_mediapipe_landmarker()

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
    preview_timestamp_ms = 0
    frame_idx = 0
    last_preview_landmarks = None

    print("Cámara iniciada. C=grabar, S=detener y predecir, Q=salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if recording:
            frames_buffer.append(frame.copy())

        if frame_idx % LIVE_DETECT_EVERY == 0:
            last_preview_landmarks, preview_timestamp_ms = _detect_mediapipe(
                preview_landmarker, frame, preview_timestamp_ms, frame_ms=40 * LIVE_DETECT_EVERY
            )

        display = draw_landmarks_points(frame.copy(), last_preview_landmarks)
        cv2.imshow(MAIN_WINDOW, draw_overlay(display, recording, last_result))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and not recording:
            recording, frames_buffer, last_result = True, [], None
            print("-> Grabando toma de prueba...")

        elif key == ord('s') and recording:
            recording = False
            print(f"-> Prediciendo sobre {len(frames_buffer)} frames...")

            frame_medio = frames_buffer[len(frames_buffer) // 2] if frames_buffer else None
            result, timestamp_ms = predict_clip(model, landmarker, frames_buffer, class_names, max_frames, timestamp_ms)
            frames_buffer = []

            if not result["valid"]:
                print(f"  [TOMA NO VÁLIDA] {result['reason']}")
                last_result = None
                continue

            print(f"  Cara detectada en {result['detected']}/{result['total']} frames")

            if not result["es_frase_de_riesgo"]:
                # No es ninguna de las 57 frases conocidas (ej. dijiste "hola",
                # "adiós", algo normal) -- no se guarda captura, no se alerta,
                # no se hace nada. El modelo SIEMPRE tiene que elegir una clase
                # internamente, pero acá se descarta por baja confianza/margen.
                print(f"  -> No es ninguna frase de riesgo conocida (más parecido: "
                      f"'{result['word']}' con {result['prob']*100:.1f}%, pero no supera el umbral) "
                      "-- no se hace nada.")
                last_result = None
                continue

            last_result = (result["word"], result["prob"])
            print(f"  -> Predicción: '{result['word']}' con {result['prob']*100:.1f}% de confianza")

            if frame_medio is not None:
                landmarks_px, timestamp_ms = detect_frame_landmarks(landmarker, frame_medio, timestamp_ms)
                capture_path = save_capture(frame_medio, landmarks_px, result["word"], result["prob"])
                print(f"  Captura guardada: {capture_path}")
            print(f"  Probabilidades: { {k: round(v,3) for k,v in result['all_probs'].items()} }")

        elif key == ord('q'):
            break

    landmarker.close()
    preview_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
