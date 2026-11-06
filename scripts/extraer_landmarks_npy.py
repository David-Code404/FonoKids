"""
extraer_landmarks_npy.py
----------------------------
Único script de extracción de landmarks del proyecto -- recorre los clips
grabados (data/sesiones_continuas/<palabra>/clips/*.avi), corre un detector
de landmarks faciales (MediaPipe o HRNet/FAN, ver DETECTOR_BACKEND más
abajo), y por cada frame calcula un vector con posición + velocidad +
aceleración de los landmarks de labios:

    [ Posición (80) | Velocidad (80) | Aceleración (80) ]

Posición: 40 landmarks de labios (FACEMESH_LIPS) x 2 coordenadas (x, y),
normalizados con invarianza a:
    - Traslación: se centra respecto al punto medio entre los ojos.
    - Rotación: se corrige el "roll" de la cabeza usando el ángulo entre
      los ojos, así da igual si la cabeza está un poco inclinada.
    - Escala: se divide por la distancia interocular (+ épsilon 1e-6 para
      evitar división por cero/NaN si los ojos quedan pegados), así da
      igual si estás más cerca o más lejos de la cámara.

Velocidad y aceleración: derivadas discretas de la posición normalizada
(Delta1 = pos[t] - pos[t-1], Delta2 = Delta1[t] - Delta1[t-1]), agregan
información de DINÁMICA del movimiento de labios, no solo forma estática
-- el frame 0 de velocidad y aceleración se define como cero.

CONTINUIDAD TEMPORAL ESTRICTA: la longitud T de cada clip se preserva
siempre igual a la cantidad real de frames del video (no se saltea ni
desfasa tiempo). Si un frame no detecta cara:
    - Si son los primeros frames del clip: se rellenan hacia atrás
      (bfill) con la primera posición válida detectada.
    - Si son frames intermedios o del final: se rellenan hacia adelante
      (ffill) con la última posición válida.
    - Si NINGÚN frame del clip detecta cara: el clip se descarta y se
      informa por consola (no se puede rellenar la nada).

Cada clip se guarda como un .npy con forma (T, 240), en carpetas separadas
por palabra -- así no hace falta cargar todo en RAM durante el entrenamiento.

Este archivo también expone las funciones de detección (build_landmarker,
detect_frame_landmarks, LIP_INDICES, get_mouth_crop) que usan grabar_video_
continuo.py y probar_modelo.py -- es el único lugar del proyecto que habla
directo con el detector de landmarks. server/main.py sigue usando MediaPipe
directo a propósito (el modelo viejo de recortes de boca no depende del
esquema de puntos, ver nota en DETECTOR_BACKEND).

Uso:
    python extraer_landmarks_npy.py            # procesa TODAS las palabras
    python extraer_landmarks_npy.py <palabra>   # procesa solo esa palabra

Lee de:    RUTA_CLIPS/<palabra>/clips/*.avi       (ver RUTA_CLIPS más abajo)
Escribe en: data/landmarks_npy/<palabra>/*.npy      -- forma (T, LANDMARK_DIM)
           data/landmarks_npy/label_map.json
"""
import os
import sys
import json
import math
import urllib.request

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
# CAMBIÁ ESTA RUTA si querés sacar landmarks de otro dataset (ej. uno
# armado a mano en otra carpeta, o un pendrive). Por defecto apunta al
# dataset normal de esta PC (data/sesiones_continuas) -- el mismo que
# escribe scripts/grabar_video_continuo.py y lee server/main.py.
# =====================================================================
RUTA_CLIPS = os.path.join(BASE_DIR, "data", "sesiones_continuas")

# =====================================================================
# 100% detección verificada en este dataset) o "hrnet" (red tipo FAN/HRNet
# vía el paquete `face-alignment`, GPU, esquema iBUG de 68 puntos con 20 de
# labios -- más pesado, requiere GPU, y da MENOS puntos de labios que
# MediaPipe, pero con mayor precisión sub-píxel por punto según benchmarks
# académicos). Los dos backends producen esquemas de puntos DISTINTOS --
# si cambiás esto tenés que volver a extraer TODO el dataset, no se pueden
# mezclar clips extraídos con uno y con el otro.
# =====================================================================
DETECTOR_BACKEND = "hrnet"

# Cuántos frames se apilan juntos en cada sub-lote al detectar caras con
# HRNet en GPU (ver detect_landmarks_batch más abajo). Confirmado en
# producción: mandar clips largos (150+ frames, típico del modo vigilancia
# de la app web) como un solo tensor puede COLGAR el driver de CUDA/WDDM en
# Windows en vez de tirar un error limpio de memoria -- bajar este número
# evita ese cuelgue. Subilo solo si tenés una GPU con mucha más VRAM.
# Bajado de 8 a 4: el cuelgue de 90s+ siguió pasando de forma intermitente
# incluso con 8 + resize a 640px -- cada pedido más chico a la GPU reduce
# la chance de disparar ese cuelgue del driver (a costa de más overhead por
# sub-lote, pero eso es rápido comparado con un cuelgue de 90s).
HRNET_BATCH_CHUNK_SIZE = 4

OUT_DIR = os.path.join(BASE_DIR, "data", "landmarks_npy")
LABEL_MAP_PATH = os.path.join(OUT_DIR, "label_map.json")

MEDIAPIPE_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

MIN_DETECTION_CONFIDENCE = 0.7
ROI_SIZE = 112
MOUTH_MARGIN_RATIO = 1.3

# Mismo umbral que usa el servidor en vivo para rechazar una toma real
# (MIN_DETECTION_RATE_VALID en server/main.py) -- antes acá solo se
# descartaba un clip si NINGÚN frame detectaba cara, así que clips con
# "cara en 1/29 frames" (3%) igual entraban al dataset de entrenamiento,
# rellenados casi enteros por bfill/ffill. Eso entrena al modelo con datos
# peores de los que la app aceptaría de un usuario real -- entrenar con el
# mismo estándar de calidad que se exige en producción.
MIN_DETECTION_RATE_TRAIN = 0.6

MAX_CAMERAS_TO_CHECK = 5  # cuántos índices probar al buscar cámaras conectadas

# Épsilon para evitar división por cero/NaN al normalizar por la distancia
# interocular (ej. si el detector devuelve landmarks degenerados).
SCALE_EPSILON = 1e-6

if DETECTOR_BACKEND == "hrnet":
    # Esquema iBUG/300W de 68 puntos (el que usa face-alignment/HRNet):
    # boca = índices 48-67 (12 del contorno exterior + 8 del interior).
    LIP_INDICES = list(range(48, 68))          # 20 puntos de labios
    # Comisuras de la boca (48 = derecha, 54 = izquierda) -- reemplazan a
    # los ojos como referencia de normalización (ver más abajo, el porqué
    # del cambio). Son las dos puntas del contorno exterior de labios.
    MOUTH_RIGHT_CORNER = 48
    MOUTH_LEFT_CORNER = 54
else:
    # Índices de labios (interior+exterior) del esquema de 468 puntos de
    # MediaPipe FaceMesh (constante estándar FACEMESH_LIPS -- hardcodeada
    # porque esta versión de mediapipe (1.0.1, solo API "tasks") no expone
    # el módulo "solutions" viejo).
    LIP_INDICES = sorted({
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        185, 40, 39, 37, 0, 267, 269, 270, 409,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        191, 80, 81, 82, 13, 312, 311, 310, 415,
    })
    # Comisuras de la boca en el esquema MediaPipe (61 = derecha, 291 =
    # izquierda) -- ya están incluidas en LIP_INDICES de arriba.
    MOUTH_RIGHT_CORNER = 61
    MOUTH_LEFT_CORNER = 291

# Cantidad final de números por frame: posición + velocidad + aceleración
# de los landmarks de labios. Depende de cuántos puntos de labios tenga el
# backend activo (40 en MediaPipe, 20 en HRNet). Lo usan lip_reading_
# dataset.py / train_landmarks_transformer.py / probar_modelo.py -- aunque
# ellos igual detectan la dimensión real del .npy, no confían solo en esto.
POSITION_DIM = len(LIP_INDICES) * 2
LANDMARK_DIM = POSITION_DIM * 3


# =====================================================================
# DETECTOR: construcción y detección, abstraídas sobre los dos backends
# posibles (MediaPipe / HRNet-FAN) -- el resto del proyecto SIEMPRE debe
# usar build_landmarker() + detect_frame_landmarks(), nunca llamar directo
# a la API de MediaPipe o face_alignment, para que cambiar DETECTOR_BACKEND
# acá arriba alcance para cambiarlo en todo el pipeline (extracción,
# grabación en vivo, prueba del modelo).
# =====================================================================
def build_landmarker():
    """Construye el detector según DETECTOR_BACKEND. Devuelve un objeto
    opaco -- pasarlo tal cual a detect_frame_landmarks()."""
    if DETECTOR_BACKEND == "hrnet":
        return _build_hrnet_detector()
    return _build_mediapipe_landmarker()


def detect_landmarks_batch(landmarker, frames_bgr, timestamp_ms=0, frame_ms=40):
    """Versión en LOTE de detect_frame_landmarks: procesa un clip entero de
    una sola vez en vez de frame por frame -- esto es lo que hace que
    probar_modelo.py y el /predict del servidor tarden mucho menos con HRNet.

    Con HRNet, la parte más pesada es la DETECCIÓN de cara (no la regresión
    de los 68 puntos en sí). Acá se manda el clip entero al GPU como un solo
    batch para detectar todas las caras de una, en vez de repetir la
    detección completa por separado en cada uno de los ~25-30 frames por
    segundo de clip. Después sí se recorre frame por frame para sacar los
    landmarks finales (la librería `face-alignment` no expone esa parte en
    batch), pero esa etapa es mucho más liviana.

    Con MediaPipe no hay ganancia real de batchear (ya corre rápido y su
    modo VIDEO es inherentemente secuencial, necesita timestamps crecientes
    frame a frame) -- ahí simplemente se llama a detect_frame_landmarks()
    en un loop, exactamente igual que antes.

    Devuelve: lista de (landmarks_px o None), en el mismo orden que
    frames_bgr -- mismo resultado que llamar detect_frame_landmarks() en un
    loop, solo que más rápido con HRNet.
    """
    if DETECTOR_BACKEND != "hrnet" or not frames_bgr:
        resultados = []
        for frame in frames_bgr:
            landmarks_px, timestamp_ms = detect_frame_landmarks(landmarker, frame, timestamp_ms, frame_ms)
            resultados.append(landmarks_px)
        return resultados

    import torch

    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

    # CRÍTICO: mandar el clip ENTERO como un solo tensor a la GPU (como se
    # hacía antes) puede pedir DECENAS de GB en clips largos (ej. modo
    # vigilancia, 150-190 frames) -- en una GPU de 6GB, ese pedido tan
    # desproporcionado no siempre tira un error limpio de "out of memory":
    # confirmado en producción, el driver de CUDA/WDDM en Windows se CUELGA
    # (nunca vuelve, nunca lanza excepción) en vez de fallar. Por eso se
    # procesa en sub-lotes chicos (HRNET_BATCH_CHUNK_SIZE) -- cada pedido a
    # la GPU queda acotado a un tamaño razonable, así nunca se dispara ese
    # cuelgue del driver, además de evitar el OOM normal en GPUs chicas.
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
                # Incluso el sub-lote fue demasiado (clip de resolución muy
                # alta) -- reintentamos ese sub-lote de a un frame por vez,
                # sin perder lo ya detectado ni abortar el clip entero.
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
        # Cualquier otra falla (ej. video con resoluciones distintas por
        # frame) -- caemos a frame por frame para todo el clip, más lento
        # pero el resultado sigue siendo correcto.
        print(f"[AVISO] Deteccion en batch fallo ({e}), usando frame por frame.")
        detected_faces = None

    # REGRESIÓN de los 68 puntos: frame por frame con la API de alto nivel
    # de la librería. Se intentó batchear esto también (como la detección de
    # arriba), pero con torch.compile activado y sin Triton en Windows, un
    # tamaño de lote nuevo dispara OTRO intento de compilación fallido por
    # cada tamaño distinto que aparece -- probado: 59 frames tardaron 108s
    # en vez de mejorar. Se deja como estaba (esta etapa ya es la liviana,
    # la pesada es la detección de arriba, que sí está bien batcheada).
    resultados = []
    for i, frame_rgb in enumerate(frames_rgb):
        if detected_faces is None:
            # El batch de detección falló entero -- dejamos que detecte
            # solo, frame a frame.
            preds = landmarker.get_landmarks_from_image(frame_rgb)
        else:
            faces = detected_faces[i] if i < len(detected_faces) else []
            if not len(faces):
                # Probado y descartado: forzar el frame completo como cuadro
                # cuando SFD no encuentra cara (ej. recorte muy cerrado, sin
                # frente ni ojos) NO rescata el frame -- el regresor FAN
                # igual devuelve 68 puntos (no sabe decir "no hay nada acá"),
                # pero apretujados cerca de donde esperaría la cara completa,
                # lejos de la boca real. Es peor que descartar: parece un
                # dato válido y es basura. Mejor tratarlo como "no
                # detectado" de verdad, que rellene por bfill/ffill.
                resultados.append(None)
                continue
            preds = landmarker.get_landmarks_from_image(frame_rgb, detected_faces=faces)
        resultados.append(preds[0].astype(np.float32) if preds else None)
    return resultados


def detect_frame_landmarks(landmarker, frame_bgr, timestamp_ms, frame_ms=40):
    """Detecta landmarks en UN frame (BGR, como lo da OpenCV/cv2.VideoCapture).

    Devuelve (landmarks_px, next_timestamp_ms):
        landmarks_px: array (N, 2) en píxeles, o None si no detectó cara.
                      N y el significado de cada índice dependen de
                      DETECTOR_BACKEND (ver LIP_INDICES/MOUTH_RIGHT_CORNER/
                      MOUTH_LEFT_CORNER).
        next_timestamp_ms: para MediaPipe hay que seguir pasándolo al
                      siguiente frame (exige timestamps crecientes); HRNet
                      no lo necesita pero se devuelve igual para que el
                      código que llama no tenga que saber qué backend está
                      activo.
    """
    if DETECTOR_BACKEND == "hrnet":
        return _detect_hrnet(landmarker, frame_bgr), timestamp_ms
    return _detect_mediapipe(landmarker, frame_bgr, timestamp_ms, frame_ms)


# --- Backend MediaPipe -------------------------------------------------
def _ensure_mediapipe_model_downloaded():
    if os.path.exists(MEDIAPIPE_MODEL_PATH):
        return
    print(f"Descargando modelo de MediaPipe ({MEDIAPIPE_MODEL_URL})...")
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
    print(f"Guardado en: {MEDIAPIPE_MODEL_PATH}")


def _build_mediapipe_landmarker():
    """Intenta GPU (delegate de MediaPipe Tasks); si esta build de MediaPipe
    para Windows no lo tiene compilado, cae a CPU automáticamente y avisa."""
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
    """Convierte el resultado crudo de MediaPipe a un array (478, 2) en
    píxeles. Sigue expuesta por compatibilidad, pero el código nuevo debería
    usar detect_frame_landmarks() en vez de llamar a MediaPipe directo."""
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


# --- Backend HRNet / FAN (paquete `face-alignment`) --------------------
def _build_hrnet_detector():
    """Detector tipo HRNet: usa el paquete `face-alignment` (red FAN, la
    implementación GPU/PyTorch de referencia para landmarks faciales de
    alta fidelidad -- no es literalmente el repo original de HRNet-Facial-
    Landmark-Detection, que requiere clonar su código y bajar pesos de
    links externos inestables; esta es la vía estable/instalable con pip
    que da resultados del mismo nivel académico)."""
    import face_alignment
    import torch
    import torch._dynamo

    # face_alignment intenta compilar su red con torch.compile (backend
    # "inductor") en la primera llamada. En Windows no hay build oficial de
    # Triton, así que esa compilación SIEMPRE falla -- sin esto, la falla
    # tira una excepción en vez de caer a modo eager (que funciona perfecto,
    # solo un poco más lento en la primerísima llamada).
    torch._dynamo.config.suppress_errors = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[AVISO] No hay GPU disponible -- HRNet/FAN en CPU va a ser MUY lento "
              "(mucho más que MediaPipe en CPU). Se recomienda GPU para este backend.")

    # compile=False: face-alignment 1.5.0 compila su red con torch.compile
    # por defecto -- en Windows (sin build de Triton) ese warm-up no tira
    # excepción, se queda COLGADO para siempre en vez de caer a modo eager
    # (confirmado en server/main.py, mismo bug). Con compile=False corre
    # directo en modo eager, sin ese paso.
    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=device, flip_input=False, compile=False,
    )
    print(f">>> Detector HRNet/FAN corriendo en {device.upper()}. <<<")
    return detector


def _detect_hrnet(landmarker, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preds = landmarker.get_landmarks(rgb)  # lista de (68, 2) por cara, o None
    if not preds:
        return None
    return preds[0].astype(np.float32)


# =====================================================================
# Recorte de boca (usado por grabar_video_continuo.py y server/main.py
# para el pipeline de imágenes recortadas -- independiente de LANDMARK_DIM)
# =====================================================================
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


# =====================================================================
# Normalización de landmarks de labios (invariante a traslación/rotación/
# escala) -- posición cruda, SIN dinámica todavía.
# =====================================================================
def normalize_lip_landmarks(landmarks_px):
    """landmarks_px: array (478, 2) en pixeles. Devuelve un vector (80,)
    con los 40 puntos de labios normalizados (traslación+rotación+escala),
    o None si algo sale mal (ej. comisuras casi superpuestas -> escala ~0).

    Normaliza contra las COMISURAS DE LA BOCA (antes se usaba la distancia
    entre los ojos) -- pedido explícito: si la cámara queda encuadrada muy
    cerca (algo muy común grabando en selfie), los ojos se salen del cuadro
    y el detector de cara fallaba en el 90%+ de los frames, aunque la boca
    se viera perfecto. Usando la boca misma como referencia, el encuadre de
    los ojos deja de importar -- solo hace falta que la boca esté visible,
    que es lo único que en verdad usa el modelo."""
    right_corner = landmarks_px[MOUTH_RIGHT_CORNER]
    left_corner = landmarks_px[MOUTH_LEFT_CORNER]

    mouth_center = (right_corner + left_corner) / 2.0
    mouth_vector = left_corner - right_corner
    scale = np.linalg.norm(mouth_vector)
    if scale < 1e-3:
        # Degenerado (comisuras casi superpuestas) -- se trata como "no
        # detectado" para que se rellene por bfill/ffill en vez de meter ruido.
        return None
    angle = math.atan2(mouth_vector[1], mouth_vector[0])

    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

    lip_pts = landmarks_px[LIP_INDICES].astype(np.float64)          # (40, 2)
    translated = lip_pts - mouth_center                              # traslación
    rotated = translated @ rotation.T                                # rotación
    normalized = rotated / (scale + SCALE_EPSILON)                   # escala (+ épsilon)

    return normalized.astype(np.float32).flatten()  # (80,)


# Salto máximo plausible entre dos frames CONSECUTIVOS (en unidades
# normalizadas por distancia interocular). Un salto más grande que esto casi
# seguro es un glitch del detector (ej. enganchó mal un punto por un
# instante), no un movimiento real de labios -- se descarta como si no se
# hubiera detectado, y bfill/ffill + suavizado lo tapan.
OUTLIER_MAX_JUMP = 0.15


def is_outlier(vector, last_valid_vector):
    """True si `vector` saltó de forma implausible respecto al último frame
    válido -- indica un glitch de detección, no movimiento real."""
    if last_valid_vector is None:
        return False
    return float(np.abs(vector - last_valid_vector).max()) > OUTLIER_MAX_JUMP


# =====================================================================
# Continuidad temporal: bfill/ffill sobre la lista de posiciones del clip
# =====================================================================
def fill_missing_frames(positions):
    """positions: lista de longitud T, cada elemento es un array (80,) o
    None (frame sin cara detectada). Devuelve (T, 80) SIN ningún None:

        - bfill: los frames iniciales sin detección se rellenan con la
          primera posición válida (no se desfasa el tiempo, y no hay
          ningún frame "anterior" real con el que interpolar).
        - Huecos EN EL MEDIO (con un frame válido antes y otro después):
          interpolación LINEAL entre esos dos puntos -- más preciso que
          copiar un solo valor, porque asume que la boca se movió gradual
          entre A y B, en vez de quedarse congelada.
        - ffill: los huecos al FINAL (sin ningún frame válido después) se
          rellenan con la última posición válida -- no hay "después" real
          con el que interpolar, copiar es lo único posible.

    Devuelve None si NINGÚN frame del clip tiene una posición válida
    (no hay nada de donde rellenar)."""
    primeros_validos = [i for i, p in enumerate(positions) if p is not None]
    if not primeros_validos:
        return None

    primer_valido = primeros_validos[0]
    ultimo_valido_idx = primeros_validos[-1]
    relleno = list(positions)  # copia, no mutamos el original

    # bfill: todo lo anterior al primer frame válido copia ESE valor.
    for i in range(primer_valido):
        relleno[i] = relleno[primer_valido]

    # Interpolación lineal para huecos entre el primer y el último válido.
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

    # ffill: los huecos después del último frame válido copian ese valor.
    for i in range(ultimo_valido_idx + 1, len(relleno)):
        relleno[i] = relleno[ultimo_valido_idx]

    return np.stack(relleno, axis=0).astype(np.float32)  # (T, 80)


# Peso del suavizado exponencial (EMA): más chico = más suave (pero más
# "atraso" respecto al movimiento real). 0.4 amortigua el jitter frame a
# frame del detector sin borrar el movimiento real de la boca.
SMOOTHING_ALPHA = 0.4


def smooth_positions(positions, alpha=SMOOTHING_ALPHA):
    """Suavizado exponencial (EMA) sobre la posición YA continua (sin None,
    ver fill_missing_frames), aplicado ANTES de calcular velocidad/
    aceleración -- derivar amplifica el ruido/jitter del detector, así que
    suavizar la posición primero da una dinámica mucho más limpia."""
    smoothed = np.empty_like(positions)
    smoothed[0] = positions[0]
    for t in range(1, positions.shape[0]):
        smoothed[t] = alpha * positions[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed.astype(np.float32)


# =====================================================================
# Dinámica: velocidad (Delta1) y aceleración (Delta2) de la posición
# =====================================================================
def add_dynamics(positions):
    """positions: (T, 80), ya continuo (sin None, ver fill_missing_frames).
    Devuelve (T, 240) = concat([posición, velocidad, aceleración], eje=1).

    Delta1[0] y Delta2[0] (primer frame) se definen como cero -- no hay
    frame anterior con el que calcular una diferencia."""
    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]  # Delta1 = pos[t] - pos[t-1]

    acceleration = np.zeros_like(positions)
    acceleration[1:] = velocity[1:] - velocity[:-1]  # Delta2 = Delta1[t] - Delta1[t-1]

    return np.concatenate([positions, velocity, acceleration], axis=1).astype(np.float32)  # (T, 240)


def process_clip(landmarker, video_path, start_timestamp_ms):
    """Procesa un clip entero preservando la longitud T EXACTA (un frame
    del video = un frame de salida, sin saltear ni desfasar tiempo).

    Devuelve (sequence, detected, total, next_timestamp_ms):
        sequence: (T, 240) float32, o None si NINGÚN frame detectó cara
                  (clip descartable).
        detected: cantidad de frames donde SÍ se detectó cara (antes de
                  rellenar) -- para reportar calidad de la toma.
        total: cantidad total de frames del clip.

    El timestamp tiene que seguir subiendo durante toda la vida del
    landmarker (no reiniciarse por clip), por eso se recibe y se devuelve.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_ms = max(int(1000 / fps), 1)

    positions = []  # (80,) o None por frame -- SIN descartar ninguno
    detected, total = 0, 0
    last_valid_vector = None  # para chequeo de outliers, no se usa como relleno
    timestamp_ms = start_timestamp_ms

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1

        landmarks_px, timestamp_ms = detect_frame_landmarks(landmarker, frame, timestamp_ms, frame_ms)
        vector = normalize_lip_landmarks(landmarks_px) if landmarks_px is not None else None

        if vector is not None and is_outlier(vector, last_valid_vector):
            # Glitch de detección (salto implausible) -- se trata como "no
            # detectado" para que bfill/ffill lo rellene en vez de meter
            # ruido en la secuencia.
            vector = None

        if vector is not None:
            detected += 1
            last_valid_vector = vector
        positions.append(vector)  # mantiene la posición del frame SIEMPRE

    cap.release()
    next_timestamp_ms = timestamp_ms + frame_ms * 5

    filled = fill_missing_frames(positions)
    if filled is None:
        return None, detected, total, next_timestamp_ms

    smoothed = smooth_positions(filled)
    sequence = add_dynamics(smoothed)  # (T, 240)
    return sequence, detected, total, next_timestamp_ms


# =====================================================================
# Selección de cámara -- compartida por grabar_video_continuo.py y
# probar_modelo.py, así los dos scripts detectan/eligen cámara igual.
# =====================================================================
def open_camera(index):
    """Abre una cámara con el backend correcto según el sistema operativo.

    En Windows, cv2.VideoCapture(index) SIN backend explícito a veces elige
    un backend equivocado (ej. "obsensor", pensado para cámaras de
    profundidad) que tira 'Camera index out of range' incluso con webcams
    normales conectadas -- por eso acá se fuerza DirectShow (CAP_DSHOW),
    que es el backend estándar y confiable para webcams en Windows."""
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

        # Salteamos clips que YA tienen su .npy (de una corrida anterior) --
        # sin esto, cada vez que grabás más datos y volvés a correr este
        # script, reprocesa TODOS los clips viejos de nuevo con HRNet (lento,
        # minutos por clip cuando la GPU anda floja), no solo los nuevos.
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

            # Pedido explícito: no descartar NINGÚN clip que tenga al menos
            # un frame con cara detectada -- el umbral del 60%
            # (MIN_DETECTION_RATE_TRAIN) se probó y tiraba la gran mayoría
            # del dataset ya grabado. Se guarda igual, avisando la tasa de
            # detección real, y se rellena por bfill/ffill como siempre.
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

    # FaceAlignment (backend HRNet/FAN) no tiene .close() -- solo el
    # landmarker de MediaPipe lo necesita. hasattr() en vez de un try/except
    # silencioso, para no esconder un error real si algún día sí lo tiene y
    # falla por otro motivo.
    if hasattr(landmarker, "close"):
        landmarker.close()
    print(f"\n=== TOTAL: {total_ok} guardados | {total_fail} descartados (sin cara en ningún frame) ===")
    print(f"Dataset en: {OUT_DIR}")


if __name__ == "__main__":
    main()
