"""
probar_modelo.py
-----------------
Prende la cámara, grabás una toma (C/S) y el modelo entrenado
(models/best.pth) te dice qué frase cree que dijiste.

Arquitectura TCN + Conformer, detección con HRNet/FAN (MediaPipe solo para
el overlay en vivo, cosmético). El vector de entrada es posición +
velocidad + aceleración de los 20 puntos de labios del esquema HRNet
(120 = 40 posición + 40 velocidad + 40 aceleración).

AUTOCONTENIDO: no importa nada de otros archivos del proyecto.

Controles:
    C -> Empieza a grabar
    S -> Detiene y predice
    Q -> Sale

Uso:
    python probar_modelo.py
"""
import math
import os
import sys
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn

torch.cuda.empty_cache()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pth")
CAPTURAS_DIR = os.path.join(BASE_DIR, "capturas")

FRAME_SIZE = (640, 480)
MIN_FRAMES_VALID = 15
MAX_FRAMES_BUFFER = 150
MIN_DETECTION_RATE_VALID = 0.6
MAIN_WINDOW = "FonoKids - Probar modelo"

MIN_RISK_PROB = 0.40
MIN_RISK_MARGIN = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# DETECCIÓN HRNet/FAN (real, la que ve el modelo) -- esquema iBUG/300W de
# 68 puntos: boca = índices 48-67 (20 puntos), ojos = 36 (der.) y 45 (izq.)
# =====================================================================
LIP_INDICES = list(range(48, 68))
RIGHT_EYE_OUTER = 36
LEFT_EYE_OUTER = 45
SCALE_EPSILON = 1e-6
HRNET_BATCH_CHUNK_SIZE = 4  # valor probado toda la sesión sin trabar la PC -- NO
# subirlo sin probarlo antes: un sub-lote más grande manda más al GPU de una sola
# vez, y en esta PC eso es justo lo que disparó cuelgues del driver de NVIDIA.


def build_hrnet_detector():
    """Detector tipo HRNet vía el paquete `face-alignment` (red FAN)."""
    import face_alignment
    import torch._dynamo

    # face_alignment intenta compilar su red con torch.compile en la primera
    # llamada -- en Windows no hay build oficial de Triton, así que esa
    # compilación SIEMPRE falla. Sin esto tira una excepción en vez de caer
    # a modo eager (que funciona perfecto, solo un poco más lento la primera vez).
    torch._dynamo.config.suppress_errors = True

    if DEVICE == "cpu":
        print("[AVISO] No hay GPU disponible -- HRNet/FAN en CPU va a ser MUY lento.")

    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=DEVICE, flip_input=False,
    )
    print(f">>> Detector HRNet/FAN corriendo en {DEVICE.upper()}. <<<")
    return detector


def detect_hrnet(detector, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preds = detector.get_landmarks(rgb)  # lista de (68, 2) por cara, o None
    if not preds:
        return None
    return preds[0].astype(np.float32)


def detect_landmarks_batch(detector, frames_bgr):
    """Detecta landmarks en un clip entero: detección de cara en sub-lotes
    (GPU), regresión de los 68 puntos frame por frame -- ver el porqué de
    los sub-lotes en el comentario de HRNET_BATCH_CHUNK_SIZE."""
    if not frames_bgr:
        return []

    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    device = detector.face_detector.device
    chunk_size = HRNET_BATCH_CHUNK_SIZE
    detected_faces = []
    try:
        for start in range(0, len(frames_rgb), chunk_size):
            chunk = frames_rgb[start:start + chunk_size]
            batch = torch.stack([
                torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1) for f in chunk
            ]).to(device)
            try:
                chunk_faces = detector.face_detector.detect_from_batch(batch)
            except torch.cuda.OutOfMemoryError:
                del batch
                print(f"[AVISO] Sub-lote de {len(chunk)} frames sin memoria, reintentando de a un frame.")
                chunk_faces = []
                for frame in chunk:
                    single = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0).to(device)
                    chunk_faces.extend(detector.face_detector.detect_from_batch(single))
                    del single
            detected_faces.extend(chunk_faces)
            del batch
    except Exception as e:
        print(f"[AVISO] Deteccion en batch fallo ({e}), usando frame por frame.")
        detected_faces = None

    resultados = []
    for i, frame_rgb in enumerate(frames_rgb):
        if detected_faces is None:
            preds = detector.get_landmarks_from_image(frame_rgb)
        else:
            faces = detected_faces[i] if i < len(detected_faces) else []
            if not len(faces):
                resultados.append(None)
                continue
            preds = detector.get_landmarks_from_image(frame_rgb, detected_faces=faces)
        resultados.append(preds[0].astype(np.float32) if preds else None)
    return resultados


def normalize_lip_landmarks(landmarks_px):
    """landmarks_px: array (68, 2) en píxeles (esquema HRNet). Devuelve un
    vector (40,) con los 20 puntos de labios normalizados (traslación +
    rotación + escala), o None si algo sale mal (ej. ojos muy pegados)."""
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

    lip_pts = landmarks_px[LIP_INDICES].astype(np.float64)
    translated = lip_pts - eye_center
    rotated = translated @ rotation.T
    normalized = rotated / (scale + SCALE_EPSILON)

    return normalized.astype(np.float32).flatten()


OUTLIER_MAX_JUMP = 0.15


def is_outlier(vector, last_valid_vector):
    if last_valid_vector is None:
        return False
    return float(np.abs(vector - last_valid_vector).max()) > OUTLIER_MAX_JUMP


def fill_missing_frames(positions):
    """positions: lista de (40,) o None por frame. Devuelve (T, 40) sin
    ningún None -- bfill al principio, interpolación lineal en huecos del
    medio, ffill al final. None si NINGÚN frame tuvo posición válida."""
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
    """(T, 40) -> (T, 120) = [posición | velocidad | aceleración]."""
    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]

    acceleration = np.zeros_like(positions)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    return np.concatenate([positions, velocity, acceleration], axis=1).astype(np.float32)


# =====================================================================
# Cámara
# =====================================================================
MAX_CAMERAS_TO_CHECK = 5


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
    while True:
        choice = input(f"¿Cuál querés usar? (número de índice, ej. {cameras[0]}): ").strip()
        try:
            idx = int(choice)
            if idx in cameras:
                return idx
        except ValueError:
            pass
        print(f"Opción inválida -- elegí uno de {cameras}.")


# =====================================================================
# MediaPipe -- SOLO para el overlay en vivo (rápido, cosmético) -- tiene su
# PROPIO landmarker y su propio contador de timestamp, separado del de
# HRNet (que no usa timestamps), así que no hay conflicto entre los dos.
# =====================================================================
MEDIAPIPE_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MIN_DETECTION_CONFIDENCE = 0.7

PREVIEW_LIP_INDICES = sorted({
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
})
PREVIEW_RIGHT_EYE_OUTER, PREVIEW_LEFT_EYE_OUTER = 33, 263
LIVE_DETECT_EVERY = 2  # correr el detector del preview cada N frames


def build_mediapipe_preview():
    if not os.path.exists(MEDIAPIPE_MODEL_PATH):
        print(f"Descargando modelo de MediaPipe ({MEDIAPIPE_MODEL_URL})...")
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    def _options(delegate):
        return mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH, delegate=delegate),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_DETECTION_CONFIDENCE,
        )

    try:
        landmarker = mp_vision.FaceLandmarker.create_from_options(_options(mp_python.BaseOptions.Delegate.GPU))
        print(">>> MediaPipe (preview) corriendo en GPU. <<<")
    except Exception:
        landmarker = mp_vision.FaceLandmarker.create_from_options(_options(mp_python.BaseOptions.Delegate.CPU))
        print(">>> MediaPipe (preview) corriendo en CPU. <<<")
    return landmarker


def detect_mediapipe_preview(landmarker, frame_bgr, timestamp_ms, frame_ms):
    import mediapipe as mp
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    landmarks_px = None
    if result.face_landmarks:
        face = result.face_landmarks[0]
        h, w = frame_bgr.shape[:2]
        landmarks_px = np.array([[lm.x * w, lm.y * h] for lm in face], dtype=np.float32)
    return landmarks_px, timestamp_ms + frame_ms


# =====================================================================
# Modelo: TCN + Conformer (arquitectura idéntica a train_landmarks_
# transformer.py -- tiene que coincidir para poder cargar el checkpoint)
# =====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        residual = x
        x = x.transpose(1, 2)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        x = x.transpose(1, 2)
        return self.relu(residual + self.dropout(x))


class TCNFeatureExtractor(nn.Module):
    def __init__(self, channels, n_blocks=3, kernel_size=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TCNBlock(channels, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])

    def forward(self, x, key_padding_mask=None):
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, conv_kernel_size=15, dropout=0.1):
        super().__init__()
        self.ff1 = ConformerFeedForward(d_model, dim_feedforward, dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ff2 = ConformerFeedForward(d_model, dim_feedforward, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        x = x + 0.5 * self.ff1(x)
        residual = x
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = residual + self.attn_dropout(attn_out)
        x = self.conv_module(x, key_padding_mask=src_key_padding_mask)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dim_feedforward, conv_kernel_size=15, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dim_feedforward, conv_kernel_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


def masked_mean_pool(x, src_key_padding_mask):
    if src_key_padding_mask is None:
        return x.mean(dim=1)
    real_mask = (~src_key_padding_mask).unsqueeze(-1).float()
    summed = (x * real_mask).sum(dim=1)
    counts = real_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class LipReadingConformer(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim=128,
                 n_layers=2, n_heads=8, dim_feedforward=256,
                 conv_kernel_size=15, tcn_blocks=3, tcn_kernel_size=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tcn = TCNFeatureExtractor(hidden_dim, n_blocks=tcn_blocks, kernel_size=tcn_kernel_size, dropout=dropout)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.conformer_encoder = ConformerEncoder(
            d_model=hidden_dim, n_layers=n_layers, n_heads=n_heads,
            dim_feedforward=dim_feedforward, conv_kernel_size=conv_kernel_size, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = self.tcn(x, key_padding_mask=src_key_padding_mask)
        x = self.pos_encoder(x)
        x = self.conformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        pooled = masked_mean_pool(x, src_key_padding_mask)
        return self.classifier(pooled)


def make_padding_mask(lengths, max_len, device):
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


# =====================================================================
# Carga del modelo y predicción
# =====================================================================
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
    smoothed = smooth_positions(positions)
    sequence_np = add_dynamics(smoothed)
    sequence = torch.from_numpy(sequence_np).float()

    T = sequence.shape[0]
    if T < max_frames:
        real_length = T
        padding = torch.zeros((max_frames - T,) + sequence.shape[1:], dtype=sequence.dtype)
        sequence = torch.cat([sequence, padding], dim=0)
    elif T > max_frames:
        # Recorte del MEDIO, no del principio -- si la toma duró más que
        # max_frames, quedarse con los primeros corta el FINAL de la
        # frase. NO se remuestrea por interpolación acá (a diferencia del
        # otro prototipo): este modelo usa velocidad/aceleración, que son
        # derivadas frame-a-frame -- interpolar el eje de tiempo las
        # rompería. Recortar sí es seguro, solo se pierden frames de las
        # puntas, no se corrompe la dinámica de los que quedan.
        start = (T - max_frames) // 2
        sequence = sequence[start:start + max_frames]
        real_length = max_frames
    else:
        real_length = T

    mask = make_padding_mask(torch.tensor([real_length]), max_frames, sequence.device)
    return sequence.unsqueeze(0), mask


def predict_clip(model, detector, frames, class_names, max_frames):
    """Igual que extraer_landmarks_npy.py: se guarda un valor por CADA
    frame (None si no se detectó cara) y se rellena con bfill/ffill, en vez
    de saltear frames -- así la predicción usa exactamente la misma
    continuidad temporal con la que se entrenó."""
    total = len(frames)

    if total < MIN_FRAMES_VALID:
        return {"valid": False, "reason": f"toma muy corta ({total} frames, "
                f"minimo {MIN_FRAMES_VALID}) -- probablemente no dijiste la frase completa"}

    positions = []
    detected = 0
    last_valid_vector = None

    landmarks_por_frame = detect_landmarks_batch(detector, frames)

    for landmarks_px in landmarks_por_frame:
        vector = normalize_lip_landmarks(landmarks_px) if landmarks_px is not None else None

        if vector is not None and is_outlier(vector, last_valid_vector):
            vector = None

        if vector is not None:
            detected += 1
            last_valid_vector = vector
        positions.append(vector)

    detection_rate = detected / total
    filled = fill_missing_frames(positions)
    if filled is None or detection_rate < MIN_DETECTION_RATE_VALID:
        return {"valid": False, "reason": f"cara detectada solo en {detected}/{total} frames "
                f"({detection_rate*100:.0f}%) -- encuadre malo, repetí la toma"}

    sequence, mask = preprocess_sequence(filled, max_frames)
    sequence, mask = sequence.to(DEVICE), mask.to(DEVICE)
    with torch.inference_mode():
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
    }


def draw_landmarks_points(frame, landmarks_px):
    """Dibuja los puntos de MediaPipe sobre el frame: rojo = resto de la
    malla facial (no se usa), amarillo = esquinas de los ojos (referencia
    de normalización), verde = los 40 puntos de labios."""
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
    print(f"Clases del modelo ({len(class_names)}): {class_names}")
    print(f"Modelo cargado en {DEVICE}: {MODEL_PATH}")

    print("Cargando detector de landmarks (HRNet/FAN)...")
    detector = build_hrnet_detector()

    print("Cargando MediaPipe para el preview en vivo (rápido, solo visual)...")
    preview_landmarker = build_mediapipe_preview()

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
    preview_timestamp_ms = 0
    frame_idx = 0
    last_preview_landmarks = None

    print("Cámara iniciada. C=grabar, S=detener y predecir, Q=salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if recording and len(frames_buffer) < MAX_FRAMES_BUFFER:
            frames_buffer.append(frame.copy())

        if frame_idx % LIVE_DETECT_EVERY == 0:
            last_preview_landmarks, preview_timestamp_ms = detect_mediapipe_preview(
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
            result = predict_clip(model, detector, frames_buffer, class_names, max_frames)
            frames_buffer = []

            if not result["valid"]:
                print(f"  [TOMA NO VÁLIDA] {result['reason']}")
                last_result = None
                continue

            print(f"  Cara detectada en {result['detected']}/{result['total']} frames")

            if not result["es_frase_de_riesgo"]:
                print(f"  -> No es ninguna frase de riesgo conocida (más parecido: "
                      f"'{result['word']}' con {result['prob']*100:.1f}%, pero no supera el umbral) "
                      "-- no se hace nada.")
                last_result = None
                continue

            last_result = (result["word"], result["prob"])
            print(f"  -> Predicción: '{result['word']}' con {result['prob']*100:.1f}% de confianza")

            if frame_medio is not None:
                landmarks_px = detect_hrnet(detector, frame_medio)
                capture_path = save_capture(frame_medio, landmarks_px, result["word"], result["prob"])
                print(f"  Captura guardada: {capture_path}")
            print(f"  Probabilidades: { {k: round(v,3) for k,v in result['all_probs'].items()} }")

        elif key == ord('q'):
            break

    detector = None
    preview_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
