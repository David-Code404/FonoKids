"""
grabar_video_continuo.py
---------------------------
Graba tomas cortas de una palabra, una por una, sin cerrar el programa entre
tomas. Cada C->S guarda un clip ya cortado y listo (no hace falta después
correr ningún script de recorte por silencios).

Muestra en vivo (con MediaPipe, el mismo detector que usa la extracción real)
un recuadro sobre la boca detectada + una ventana aparte con el recorte
agrandado, para que veas el encuadre ANTES de terminar de grabar, no después.

Controles:
    S -> Empieza a grabar una toma nueva
    A -> Detiene y GUARDA esa toma (el programa sigue abierto, listo para la próxima)
    Q -> Sale del programa

Salida:
    sesiones_continuas/<palabra>/clips/<palabra>_<persona>_0001.avi, 0002.avi, ...
    (la persona sirve para agrupar las grabaciones por fecha/persona en la app)

Después corré:
    python extraer_landmarks_npy.py <palabra>
"""
import os
import sys
import unicodedata

import cv2
import numpy as np

from extraer_landmarks_npy import build_landmarker, detect_frame_landmarks, LIP_INDICES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")

CAM_INDEX = 0
FPS = 25
FRAME_SIZE = (640, 480)
LIVE_DETECT_EVERY = 2      # correr MediaPipe cada N frames (rendimiento en vivo)
LIVE_PREVIEW_SIZE = 260
MAIN_WINDOW = "SpeakShadow - Grabar tomas"
MOUTH_WINDOW = "SpeakShadow - Boca (preview)"


def ask_word():
    # Normalización NFC: así "á" (precompuesto) y "a"+tilde combinante no
    # generan dos carpetas/labels distintas para la "misma" palabra.
    while True:
        word = input("\n¿Qué palabra vas a grabar?: ").strip().lower().replace(" ", "_")
        word = unicodedata.normalize("NFC", word)
        word = "".join(c for c in word if c.isalnum() or c == "_")
        if word:
            return word
        print("Vacío, probá de nuevo.")


def ask_person():
    # Sin "_" a propósito: el nombre de archivo es <palabra>_<persona>_0001.avi
    # y el servidor separa la persona del número de toma buscando el último "_".
    while True:
        person = input("¿Quién va a grabar (nombre o apodo)?: ").strip().lower().replace(" ", "-")
        person = "".join(c for c in person if c.isalnum() or c == "-")
        if person:
            return person
        print("Vacío, probá de nuevo.")


def open_camera(index):
    """Fuerza DirectShow en Windows -- sin esto, cv2.VideoCapture puede
    elegir un backend equivocado ('obsensor') que tira 'Camera index out
    of range' incluso con una webcam normal conectada y andando."""
    if sys.platform == "win32":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def count_existing_clips(clips_dir):
    if not os.path.isdir(clips_dir):
        return 0
    return len([f for f in os.listdir(clips_dir) if f.endswith(".avi")])


def compute_mouth_box(landmarks_px, frame_shape, margin_ratio=1.3):
    lip_pts = landmarks_px[LIP_INDICES]
    center = lip_pts.mean(axis=0)
    half_side = max((lip_pts.max(axis=0) - lip_pts.min(axis=0)).max() / 2 * margin_ratio, 20)
    h, w = frame_shape[:2]
    x1 = int(np.clip(center[0] - half_side, 0, w - 1))
    x2 = int(np.clip(center[0] + half_side, 0, w - 1))
    y1 = int(np.clip(center[1] - half_side, 0, h - 1))
    y2 = int(np.clip(center[1] + half_side, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_mouth_preview(frame, mouth_box, size=LIVE_PREVIEW_SIZE, recording=False):
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if mouth_box is not None:
        x1, y1, x2, y2 = mouth_box
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            canvas = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        cv2.putText(canvas, "Sin cara detectada", (10, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    border_color = (0, 0, 255) if recording else (0, 255, 255)
    cv2.rectangle(canvas, (0, 0), (size - 1, size - 1), border_color, 4)
    return canvas


def draw_overlay(frame, recording, word, n_saved, mouth_box):
    h, w = frame.shape[:2]

    status_text, color = ("GRABANDO", (0, 0, 255)) if recording else ("LISTO (S para grabar)", (0, 200, 0))
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if recording:
        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)

    if mouth_box is not None:
        x1, y1, x2, y2 = mouth_box
        box_color = (0, 0, 255) if recording else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    cv2.putText(frame, f"Palabra: {word}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(frame, f"Tomas guardadas: {n_saved}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "S: grabar | A: detener y guardar | Q: salir",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def main():
    word = ask_word()
    person = ask_person()
    clips_dir = os.path.join(SESSIONS_DIR, word, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    n_saved = count_existing_clips(clips_dir)

    cap = open_camera(CAM_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

    # Mostramos la cámara ya prendida de una, con un aviso, mientras
    # MediaPipe carga en segundo plano (eso es lo que tarda unos segundos).
    ok, frame = cap.read()
    if ok:
        loading_frame = frame.copy()
        cv2.putText(loading_frame, "Cargando MediaPipe...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.imshow(MAIN_WINDOW, loading_frame)
        cv2.waitKey(1)

    print("Cargando MediaPipe Face Landmarker (para el preview en vivo)...")
    landmarker = build_landmarker()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = None
    recording = False
    frame_idx = 0
    timestamp_ms = 0
    frame_ms = int(1000 / FPS)
    last_mouth_box = None

    print(f"\nPalabra: '{word}' (ya tenés {n_saved} tomas guardadas).")
    print("S = grabar toma | A = detener y guardar | Q = salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer la cámara.")
            break
        frame_idx += 1

        if recording and writer is not None:
            writer.write(frame)

        if frame_idx % LIVE_DETECT_EVERY == 0:
            landmarks_px, timestamp_ms = detect_frame_landmarks(
                landmarker, frame, timestamp_ms, frame_ms=frame_ms * LIVE_DETECT_EVERY
            )
            last_mouth_box = compute_mouth_box(landmarks_px, frame.shape) if landmarks_px is not None else None

        display = draw_overlay(frame.copy(), recording, word, n_saved, last_mouth_box)
        mouth_preview = build_mouth_preview(frame, last_mouth_box, recording=recording)

        cv2.imshow(MAIN_WINDOW, display)
        cv2.imshow(MOUTH_WINDOW, mouth_preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and not recording:
            out_path = os.path.join(clips_dir, f"{word}_{person}_{n_saved + 1:04d}.avi")
            writer = cv2.VideoWriter(out_path, fourcc, FPS, FRAME_SIZE)
            recording = True
            print(f"-> Grabando toma #{n_saved + 1}...")

        elif key == ord('a') and recording:
            recording = False
            writer.release()
            writer = None
            n_saved += 1
            print(f"-> Guardada toma #{n_saved}. Apretá S para grabar otra.")

        elif key == ord('q'):
            if recording and writer is not None:
                writer.release()
                n_saved += 1
                print(f"-> Toma en curso guardada como #{n_saved} antes de salir.")
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nListo. Total de tomas de '{word}': {n_saved}")
    print(f"Guardadas en: {clips_dir}")
    print(f"Ahora corré: python extraer_landmarks_npy.py {word}")


if __name__ == "__main__":
    main()
