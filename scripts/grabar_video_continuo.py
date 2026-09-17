"""
grabar_video_continuo.py
---------------------------
Graba tomas cortas de una palabra, una por una, sin cerrar el programa entre
tomas. Cada C->S guarda un clip ya cortado y listo (no hace falta después
correr ningún script de recorte por silencios).

Muestra en vivo (con MediaPipe, el mismo detector que usa la extracción real)
un recuadro sobre la boca detectada + una ventana aparte con el recorte
agrandado, para que veas el encuadre ANTES de terminar de grabar, no después.

Cada toma graba TAMBIÉN el audio (micrófono por defecto del sistema, 16kHz
mono -- la frecuencia que espera wav2vec2), guardado como .wav aparte con el
mismo nombre que el .avi. cv2.VideoWriter no graba audio (no es una
limitación de este script, es una limitación de OpenCV), por eso el audio se
graba con `sounddevice` en paralelo, no mezclado en el mismo archivo.

Controles:
    S -> Empieza a grabar una toma nueva (video + audio)
    A -> Detiene y GUARDA esa toma (el programa sigue abierto, listo para la próxima)
    Q -> Sale del programa

Salida:
    sesiones_continuas/<palabra>/clips/<palabra>_0001.avi, 0002.avi, ...
    sesiones_continuas/<palabra>/clips/<palabra>_0001.wav, 0002.wav, ...

Después corré:
    python extraer_landmarks_npy.py <palabra>
"""
import os
import unicodedata
import wave

import cv2
import numpy as np
import sounddevice as sd

from extraer_landmarks_npy import (
    _build_mediapipe_landmarker,
    _detect_mediapipe,
    LIP_INDICES,
    choose_camera,
    open_camera,
)

# El preview en vivo SIEMPRE usa MediaPipe (rápido, ~tiempo real), sin
# importar qué DETECTOR_BACKEND esté elegido para la extracción real -- acá
# el recuadro de boca es solo una ayuda visual para encuadrar mientras
# grabás, no hace falta la precisión de HRNet (que a ~3 fps trababa la
# vista en vivo y afectaba el timing de la grabación).
build_landmarker = _build_mediapipe_landmarker
detect_frame_landmarks = _detect_mediapipe

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")

FPS = 25
FRAME_SIZE = (640, 480)
LIVE_DETECT_EVERY = 2      # correr MediaPipe cada N frames (rendimiento en vivo)
LIVE_PREVIEW_SIZE = 260
MAIN_WINDOW = "SpeakShadow - Grabar tomas"
MOUTH_WINDOW = "SpeakShadow - Boca (preview)"

# 16kHz mono -- la frecuencia de muestreo que espera wav2vec2 (y la mayoría
# de los modelos de voz pre-entrenados). Grabar directo a esta frecuencia
# evita tener que resamplear después.
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "int16"  # 2 bytes por muestra -- coincide con sampwidth=2 al guardar el .wav


def save_wav(path, audio_chunks, sample_rate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS):
    """audio_chunks: lista de arrays (frames, channels) int16, tal como los
    entrega el callback de sounddevice. Si está vacía (ej. no había
    micrófono disponible), no escribe nada -- el .avi de video se guarda
    igual, solo falta el audio de esa toma."""
    if not audio_chunks:
        return False
    audio_data = np.concatenate(audio_chunks, axis=0)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return True


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
    clips_dir = os.path.join(SESSIONS_DIR, word, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    n_saved = count_existing_clips(clips_dir)

    cam_index = choose_camera()
    if cam_index is None:
        return

    cap = open_camera(cam_index)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara (índice {cam_index}).")
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

    # La cámara a veces no entrega EXACTAMENTE el tamaño pedido (más común
    # todavía con DirectShow) -- si VideoWriter se abre con un tamaño que no
    # coincide con el de los frames reales, graba en silencio un archivo
    # vacío/corrupto (sin tirar ningún error). Por eso se usa el tamaño REAL
    # del primer frame leído, no FRAME_SIZE a ciegas.
    real_frame_size = (frame.shape[1], frame.shape[0]) if ok else FRAME_SIZE
    if real_frame_size != FRAME_SIZE:
        print(f"[AVISO] La cámara entrega {real_frame_size} en vez de {FRAME_SIZE} -- "
              "se usa el tamaño real para que los clips no queden corruptos.")

    print("Cargando MediaPipe Face Landmarker (para el preview en vivo)...")
    landmarker = build_landmarker()

    # El micrófono se abre UNA sola vez al arrancar (no cada toma) -- abrir
    # y cerrar un InputStream por toma es lo que suele disparar clics/pops
    # al principio de la grabación en Windows. audio_state["on"] es lo que
    # realmente prende/apaga la captura -- el callback corre siempre en su
    # propio hilo, pero solo junta frames mientras audio_state["on"] es True.
    audio_state = {"on": False}
    audio_chunks = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[AVISO] audio: {status}")
        if audio_state["on"]:
            audio_chunks.append(indata.copy())

    try:
        audio_stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS,
            dtype=AUDIO_DTYPE, callback=audio_callback,
        )
        audio_stream.start()
        print(f"Micrófono abierto ({AUDIO_SAMPLE_RATE}Hz mono).")
    except Exception as e:
        # Sin micrófono disponible, seguimos grabando solo video -- mejor
        # perder el audio de esta sesión que no poder grabar nada.
        print(f"[AVISO] No se pudo abrir el micrófono ({e}) -- se graba SOLO video, sin audio.")
        audio_stream = None

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = None
    recording = False
    frame_idx = 0
    timestamp_ms = 0
    frame_ms = int(1000 / FPS)
    last_mouth_box = None
    lecturas_fallidas_seguidas = 0
    MAX_LECTURAS_FALLIDAS = 30  # ~1 segundo a 30fps -- ahí sí asumimos que la cámara se desconectó

    print(f"\nPalabra: '{word}' (ya tenés {n_saved} tomas guardadas).")
    print("S = grabar toma | A = detener y guardar | Q = salir")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                lecturas_fallidas_seguidas += 1
                if lecturas_fallidas_seguidas == 1:
                    print("[AVISO] La cámara falló al leer un frame -- reintentando "
                          "(la toma en curso NO se pierde)...")
                if lecturas_fallidas_seguidas >= MAX_LECTURAS_FALLIDAS:
                    print("La cámara dejó de responder por más de 1 segundo, cierro el programa.")
                    break
                cv2.waitKey(10)
                continue
            lecturas_fallidas_seguidas = 0
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
                base_name = f"{word}_{n_saved + 1:04d}"
                out_path = os.path.join(clips_dir, f"{base_name}.avi")
                audio_out_path = os.path.join(clips_dir, f"{base_name}.wav")
                writer = cv2.VideoWriter(out_path, fourcc, FPS, real_frame_size)
                if not writer.isOpened():
                    print(f"[ERROR] No se pudo crear el archivo de video en {out_path} "
                          "-- revisá que la carpeta exista y tengas permisos de escritura.")
                    writer = None
                    continue
                audio_chunks.clear()
                audio_state["on"] = audio_stream is not None
                recording = True
                print(f"-> Grabando toma #{n_saved + 1}...")

            elif key == ord('a') and recording:
                recording = False
                audio_state["on"] = False
                if writer is not None:
                    writer.release()
                    writer = None
                if save_wav(audio_out_path, audio_chunks):
                    audio_chunks.clear()
                n_saved += 1
                print(f"-> Guardada toma #{n_saved}. Apretá S para grabar otra.")

            elif key == ord('q'):
                if recording and writer is not None:
                    audio_state["on"] = False
                    writer.release()
                    writer = None
                    save_wav(audio_out_path, audio_chunks)
                    n_saved += 1
                    print(f"-> Toma en curso guardada como #{n_saved} antes de salir.")
                break
    finally:
        # Pase lo que pase (error, cámara desconectada, Ctrl+C) la toma en
        # curso se cierra bien -- si no, el .avi queda sin el header/index
        # final y no se puede leer después (esto era el bug real).
        if writer is not None:
            audio_state["on"] = False
            writer.release()
            save_wav(audio_out_path, audio_chunks)
            n_saved += 1
            print(f"-> Toma en curso guardada como #{n_saved} antes de cerrar.")
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nListo. Total de tomas de '{word}': {n_saved}")
    print(f"Guardadas en: {clips_dir} (.avi con video, .wav con audio 16kHz mono)")
    print(f"Ahora corré: python extraer_landmarks_npy.py {word}")


if __name__ == "__main__":
    main()
