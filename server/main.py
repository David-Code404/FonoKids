"""
server/main.py
---------------
API HTTP de SpeakShadow para práctica de pronunciación infantil. Recibe un
video corto (grabado desde el navegador, un chico diciendo una palabra
objetivo), corre EXACTAMENTE el mismo pipeline que scripts/probar_modelo.py
(FAN -> landmarks de labios -> TCN + Conformer) y devuelve si la pronunció
bien o mal.

El modelo se entrena con clases de a pares por palabra objetivo
("<palabra>_correcto" / "<palabra>_incorrecto") -- el nombre de la clase
ganadora ya dice CUÁL palabra se intentó decir Y si estuvo bien dicha, sin
necesitar un módulo de decisión aparte para eso (ver _parse_clase_predicha).

Todo el código de detección/modelo de acá abajo está copiado tal cual de
scripts/probar_modelo.py (ese script NO se toca) -- así el server predice
exactamente igual que la prueba con cámara que ya funciona bien.

Uso:
    python server/main.py
    (escucha solo en 127.0.0.1:8000 -- nada más que esta misma PC, sin
    exponerlo a la red local)
"""
import math
import os
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pth")  # igual que probar_modelo.py
GOAL_PER_WORD = 500

MIN_FRAMES_VALID = 15           # menos que esto = probablemente no se dijo la frase completa
MIN_DETECTION_RATE_VALID = 0.6  # menos del 60% de frames con cara detectada = toma mala

# Ancho máximo de cada frame ANTES de mandarlo a HRNet -- la cámara del
# navegador graba en Full HD (1920px), muchísimo más de lo necesario para
# detectar una cara. 640px es lo mismo que usa probar_modelo.py (FRAME_SIZE).
MAX_FRAME_WIDTH = 640

# Igual que probar_modelo.py: el modelo SIEMPRE elige una de las clases
# entrenadas (no sabe decir "no sé"), así que esto es lo que distingue "el
# modelo está seguro de esta lectura" de "está adivinando entre dos clases
# parecidas, no confiar en este resultado todavía".
MIN_CONFIDENCE_PROB = 0.40
MIN_CONFIDENCE_MARGIN = 0.15

# Convención de nombres de clase en el checkpoint: cada palabra objetivo
# aporta la clase "<palabra>_correcto" y una o más clases de error
# "<palabra>_incorrecto" o "<palabra>_incorrecto_<subtipo>" (ej.
# "perro_incorrecto_lambdacismo", "perro_incorrecto_dentalizacion",
# "perro_incorrecto_omision" -- variantes de error real de logopedia, no
# "otra palabra que suena parecido"). El propio nombre de la clase ganadora
# ya dice qué palabra se intentó, si quedó bien dicha, y (si está disponible)
# qué tipo de error fue -- sin necesitar una lista aparte tipo la vieja
# RISK_WORDS.
SUFIJO_CORRECTO = "_correcto"
MARCADOR_INCORRECTO = "_incorrecto"


def _parse_clase_predicha(nombre_clase):
    """'perro_correcto' -> ('perro', True, None).
    'perro_incorrecto' -> ('perro', False, None).
    'perro_incorrecto_lambdacismo' -> ('perro', False, 'lambdacismo').
    Si la clase no tiene ninguno de los dos marcadores (checkpoint viejo o
    mal entrenado), devuelve (nombre_clase, None, None) -- se trata como "no
    se pudo determinar corrección" en vez de asumir un valor."""
    if nombre_clase.endswith(SUFIJO_CORRECTO):
        return nombre_clase[: -len(SUFIJO_CORRECTO)], True, None

    idx = nombre_clase.find(MARCADOR_INCORRECTO)
    if idx != -1:
        palabra = nombre_clase[:idx]
        resto = nombre_clase[idx + len(MARCADOR_INCORRECTO):]
        subtipo = resto[1:] if resto.startswith("_") else None
        return palabra, False, subtipo or None

    return nombre_clase, None, None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# DETECCIÓN HRNet/FAN -- copiado de scripts/probar_modelo.py, esquema
# iBUG/300W de 68 puntos: boca = índices 48-67 (20 puntos), ojos = 36
# (der.) y 45 (izq.)
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

    # compile=False: face-alignment 1.5.0 compila su red con torch.compile
    # por defecto -- en esta PC (Windows, sin Triton) ese warm-up no tira
    # excepción, se queda COLGADO para siempre (confirmado: el arranque del
    # server nunca pasaba de "Compiling face alignment model..."). Con
    # compile=False corre directo en modo eager, sin ese paso.
    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=DEVICE, flip_input=False, compile=False,
    )
    print(f">>> Detector HRNet/FAN corriendo en {DEVICE.upper()}. <<<")
    return detector


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
# Modelo: TCN + Conformer -- copiado de scripts/probar_modelo.py
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


def safe_torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


app = FastAPI(title="SpeakShadow API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global: modelo y landmarker se cargan UNA sola vez al arrancar,
# no en cada request (cargar el modelo por request sería lentísimo).
_state = {"model": None, "landmarker": None, "class_names": None, "max_frames": None}

# =====================================================================
# TODO el trabajo de GPU (carga del modelo, precalentamiento, y CADA
# predicción) corre en el HILO PRINCIPAL del proceso -- ni un hilo aparte
# dedicado, ni el threadpool genérico de Starlette. Exactamente igual que
# scripts/probar_modelo.py (que nunca se cuelga, ni con tomas largas).
#
# Antes esto corría en un hilo dedicado aparte (siempre el mismo, pero NO
# el principal) para no bloquear el resto del servidor mientras predice.
# Confirmado en producción: los cuelgues de /predict seguían pasando igual
# ahí, cada vez más seguido -- la diferencia real entre "funciona en
# probar_modelo.py" y "se cuelga en el servidor" parece ser justo esta:
# CUDA en Windows da problemas cuando el contexto de GPU se toca desde un
# hilo que no es el principal del proceso, no importa cuán "dedicado" sea.
#
# El costo: /predict ahora bloquea TODO el servidor (ni /health responde)
# mientras predice -- pero como ya se procesaba una predicción a la vez de
# todos modos, no se pierde nada real. Y como esto bloquea el hilo que
# antes vigilaba el timeout, ya no hay forma de que el PROPIO proceso se
# reinicie solo si se cuelga -- por eso ahora hay un vigilante EXTERNO
# aparte (ver server/watchdog.sh) que chequea /health desde afuera y mata
# el proceso a la fuerza si deja de responder por mucho tiempo.
# =====================================================================


@app.on_event("startup")
async def load_everything():
    """async a propósito: FastAPI/Starlette corre los eventos de startup
    definidos como `def` (sync) en un hilo del threadpool, NO en el
    principal -- con `async def` y llamando todo directo (sin await a
    ningún threadpool), esto se ejecuta en el mismo hilo que el event loop
    de uvicorn, que ES el principal del proceso (sin --workers)."""
    if not os.path.exists(MODEL_PATH):
        print(f"[AVISO] No encontré {MODEL_PATH} -- /predict queda deshabilitado. "
              "Entrenalo con train_landmarks_transformer.py o bajalo de Colab. "
              "El resto del server (dashboard, stats) funciona igual.")
        return

    checkpoint = safe_torch_load(MODEL_PATH)
    class_names = checkpoint["class_names"]
    max_frames = checkpoint["max_frames"]
    input_dim = checkpoint.get("input_dim", 120)

    model = LipReadingConformer(num_classes=len(class_names), input_dim=input_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Cargando detector de landmarks (HRNet/FAN)...")
    landmarker = build_hrnet_detector()

    # Precalentamiento: la PRIMERA vez que HRNet corre, PyTorch intenta
    # compilar la red (torch.compile/dynamo), falla en Windows y cae a
    # modo eager -- ese intento fallido es lo que hace que la primera
    # predicción real tarde ~50-90s en vez de ~10-20s. Absorbemos ese
    # costo ACÁ, antes de aceptar requests, para que la primera
    # predicción de un usuario real ya sea rápida.
    if DEVICE == "cuda":
        try:
            print("Precalentando HRNet (primera compilación, puede tardar)...")
            dummy_frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(3)]
            detect_landmarks_batch(landmarker, dummy_frames)
            print("Precalentamiento listo.")
        except Exception as e:
            print(f"[AVISO] Precalentamiento falló ({e}), no es grave -- "
                  "la primera predicción real puede tardar más de lo normal.")

    _state.update(model=model, landmarker=landmarker, class_names=class_names, max_frames=max_frames)
    print(f"Listo. Device: {DEVICE}. Clases ({len(class_names)}): {class_names}")


@app.on_event("shutdown")
def cleanup():
    landmarker = _state["landmarker"]
    if landmarker is not None and hasattr(landmarker, "close"):
        landmarker.close()


def preprocess_sequence(positions, max_frames):
    """Copiado de scripts/probar_modelo.py -- suaviza, agrega velocidad/
    aceleración, y padea/recorta a max_frames. Recorte del MEDIO (no del
    principio) si la toma duró más que max_frames -- ver el porqué en
    probar_modelo.py. NO se remuestrea por interpolación: este modelo usa
    derivadas frame-a-frame (velocidad/aceleración), que se romperían."""
    smoothed = smooth_positions(positions)
    sequence_np = add_dynamics(smoothed)
    sequence = torch.from_numpy(sequence_np).float()

    T = sequence.shape[0]
    if T < max_frames:
        real_length = T
        padding = torch.zeros((max_frames - T,) + sequence.shape[1:], dtype=sequence.dtype)
        sequence = torch.cat([sequence, padding], dim=0)
    elif T > max_frames:
        start = (T - max_frames) // 2
        sequence = sequence[start:start + max_frames]
        real_length = max_frames
    else:
        real_length = T

    mask = make_padding_mask(torch.tensor([real_length]), max_frames, sequence.device)
    return sequence.unsqueeze(0), mask  # (1, T, N), (1, T)


@app.get("/health")
def health():
    """La app la usa para chequear que el servidor está vivo y el modelo cargado."""
    return {"status": "ok", "device": DEVICE, "classes": _state["class_names"]}


# Extensiones de video reconocidas como "clip" -- .avi es lo que graba
# grabar_video_continuo.py (dataset), .webm/.mp4/.mov es lo que graba la
# cámara del navegador (pestaña "Practicar" de la app web) cuando /predict
# guarda un intento de pronunciación con confianza suficiente (ver más abajo).
VIDEO_EXTENSIONS = (".avi", ".webm", ".mp4", ".mov")


def _count_files(folder, extensions=VIDEO_EXTENSIONS):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(extensions)])


@app.get("/dataset/stats")
def dataset_stats():
    """Progreso del dataset -- cuántos clips grabados hay por clase
    ("<palabra>_correcto" / "<palabra>_incorrecto"), para mostrar el
    dashboard en la app."""
    class_names_dirs = set()
    if os.path.isdir(SESSIONS_DIR):
        class_names_dirs.update(d for d in os.listdir(SESSIONS_DIR)
                                 if os.path.isdir(os.path.join(SESSIONS_DIR, d)))

    rows = []
    total_clips = 0
    for class_name in sorted(class_names_dirs):
        n_clips = _count_files(os.path.join(SESSIONS_DIR, class_name, "clips"))
        palabra, correcta, tipo_error = _parse_clase_predicha(class_name)
        rows.append({
            "class_name": class_name, "palabra": palabra, "correcta": correcta,
            "tipo_error": tipo_error, "clips": n_clips,
        })
        total_clips += n_clips

    return {
        "goal_per_word": GOAL_PER_WORD,
        "total_words": len(rows),
        "total_clips": total_clips,
        "words": rows,
    }


def _parse_clip_filename(class_name, filename):
    """De '<clase>_<persona>_0001.avi' saca la persona, o None si el clip
    es viejo y no tiene persona en el nombre ('<clase>_0001.avi')."""
    stem = os.path.splitext(filename)[0]
    prefix = class_name + "_"
    rest = stem[len(prefix):] if stem.lower().startswith(prefix.lower()) else stem
    if "_" in rest:
        person, _seq = rest.rsplit("_", 1)
        return person
    return None


@app.get("/dataset/recordings")
def dataset_recordings():
    """Sesiones de práctica agrupadas por fecha + clase + persona, para la
    pantalla de historial. Cada grupo incluye "thumbnail_file" -- el nombre
    del clip más reciente de ese grupo, para poder pedir una foto real del
    momento con GET /dataset/thumbnail (ver más abajo). Cada grupo también
    trae "palabra" y "correcta" (derivados del nombre de clase) para que la
    app pueda mostrar el progreso por palabra sin tener que parsear nada
    del lado del cliente."""
    if not os.path.isdir(SESSIONS_DIR):
        return {"recordings": []}

    grouped = defaultdict(list)
    for class_name in os.listdir(SESSIONS_DIR):
        clips_dir = os.path.join(SESSIONS_DIR, class_name, "clips")
        if not os.path.isdir(clips_dir):
            continue
        for filename in os.listdir(clips_dir):
            if not filename.lower().endswith(VIDEO_EXTENSIONS):
                continue
            path = os.path.join(clips_dir, filename)
            person = _parse_clip_filename(class_name, filename) or "desconocido"
            mtime = os.path.getmtime(path)
            date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            grouped[(date, class_name, person)].append((mtime, filename))

    recordings = []
    for (date, class_name, person), files in grouped.items():
        files.sort(key=lambda f: f[0], reverse=True)
        palabra, correcta, tipo_error = _parse_clase_predicha(class_name)
        recordings.append({
            "date": date,
            # Hora de la práctica más reciente del grupo -- no es una hora
            # por intento individual (acá se agrupa por día+clase+persona,
            # no clip a clip), pero alcanza para mostrar "a qué hora fue la
            # última vez" en la app.
            "time": datetime.fromtimestamp(files[0][0]).strftime("%H:%M"),
            "class_name": class_name,
            "word": palabra,
            "correcta": correcta,
            "tipo_error": tipo_error,
            "person": person,
            "count": len(files),
            "thumbnail_file": files[0][1],
        })
    recordings.sort(key=lambda r: r["date"], reverse=True)

    return {"recordings": recordings}


@app.get("/dataset/thumbnail")
def dataset_thumbnail(word: str, filename: str):
    """Devuelve un frame (JPEG) del medio de un clip guardado, para mostrar
    una foto real en el diálogo de práctica de la app en vez del placeholder.
    `word` acá es en realidad el nombre de CLASE completo (ej.
    "perro_incorrecto"), tal como lo devuelve /dataset/recordings en
    "class_name" -- se llama "word" en la URL por compatibilidad con la app.
    `filename` se valida con basename -- nunca se deja salir de clips_dir."""
    safe_filename = os.path.basename(filename)
    safe_word = os.path.basename(word)
    path = os.path.join(SESSIONS_DIR, safe_word, "clips", safe_filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Clip no encontrado.")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=422, detail="No se pudo leer el clip.")

    # OJO: NO usar CAP_PROP_FRAME_COUNT + CAP_PROP_POS_FRAMES para "saltar"
    # al frame del medio -- los .webm que graba el navegador (MediaRecorder)
    # no traen ese índice bien armado y el seek falla en silencio. Leemos
    # todo secuencialmente (igual que hace /predict, que sabemos que sí
    # funciona con estos archivos) y nos quedamos con el del medio.
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise HTTPException(status_code=422, detail="No se pudo leer ningún frame del clip.")
    frame = frames[len(frames) // 2]

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la imagen.")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


def _save_practice_clip(class_name, tmp_path, suffix):
    """Guarda una copia del clip grabado desde la app web (pestaña
    "Practicar") -- queda registrado en el historial de práctica (GET
    /dataset/recordings), igual que un clip grabado con
    grabar_video_continuo.py. Se guarda bajo el nombre de CLASE completo
    (ej. "perro_incorrecto/clips/"), no solo la palabra, para que el
    historial pueda distinguir intentos correctos de incorrectos de la
    misma palabra. Persona fija "web". Nunca tira: si falla (ej. sin
    permisos de disco), la predicción igual se devuelve normal.

    Devuelve el nombre del archivo guardado (para que /predict lo mande de
    vuelta al cliente y la app pueda mostrar la foto real al toque, sin
    esperar a la próxima carga del historial), o None si falló el guardado.
    """
    try:
        clips_dir = os.path.join(SESSIONS_DIR, class_name, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        dest_name = f"{class_name}_web_{int(time.time() * 1000)}{suffix}"
        shutil.copy(tmp_path, os.path.join(clips_dir, dest_name))
        return dest_name
    except OSError as e:
        print(f"[AVISO] No se pudo guardar el clip de práctica ({e}).")
        return None


# Ya no hace falta un semáforo acá -- al correr todo en el hilo principal
# (ver comentario grande más arriba), FastAPI ni siquiera puede empezar a
# procesar una segunda request de /predict hasta que la anterior termine,
# la serialización queda garantizada sola.

# Referencia para server/watchdog.sh (no se usa acá adentro): cuánto puede
# tardar una predicción real, sin cuelgue, en el peor caso.
PREDICT_HARD_TIMEOUT_S = 90


def _run_prediction_pipeline(tmp_path, suffix, landmarker, model, class_names, max_frames):
    """Todo el trabajo pesado (lectura de frames, HRNet, Conformer) -- código
    100% sincrónico y bloqueante a propósito, para correrlo en el hilo
    dedicado de GPU (ver _gpu_worker_loop) y no congelar el event loop de
    FastAPI mientras tarda.

    Los print() con tiempos son a propósito -- si esto se cuelga nos dice
    EXACTAMENTE en qué etapa se quedó trabado (lectura de frames, detección
    HRNet, o inferencia del modelo), en vez de tener que adivinar."""
    t_start = time.time()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="No se pudo leer el video enviado.")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # La cámara del navegador graba en Full HD (1920x1080) -- HRNet no
        # necesita esa resolución para encontrar la cara, se achica ACÁ,
        # antes de meterla al pipeline, preservando el aspecto (igual que
        # FRAME_SIZE en probar_modelo.py, que graba nativo a 640x480).
        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / w
            frame = cv2.resize(frame, (MAX_FRAME_WIDTH, round(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    print(f"[predict] frames leídos: {len(frames)} ({time.time() - t_start:.1f}s)")

    total = len(frames)
    if total == 0:
        return {
            "valid": False,
            "reason": "no se pudo leer ningún frame del video -- el formato/códec "
                      "puede no ser compatible con el servidor (probá grabar en H.264/mp4)",
        }
    if total < MIN_FRAMES_VALID:
        return {
            "valid": False,
            "reason": f"toma muy corta ({total} frames, minimo {MIN_FRAMES_VALID}) "
                      "-- probablemente no se dijo la frase completa",
        }

    landmarks_por_frame = detect_landmarks_batch(landmarker, frames)
    print(f"[predict] landmarks detectados ({time.time() - t_start:.1f}s)")

    positions = []
    detected = 0
    last_valid_vector = None
    for landmarks_px in landmarks_por_frame:
        vector = normalize_lip_landmarks(landmarks_px) if landmarks_px is not None else None
        if vector is not None and is_outlier(vector, last_valid_vector):
            vector = None  # glitch de detección, se rellena en vez de aceptarlo
        if vector is not None:
            detected += 1
            last_valid_vector = vector
        positions.append(vector)

    detection_rate = detected / total if total else 0
    filled = fill_missing_frames(positions)
    if filled is None or detection_rate < MIN_DETECTION_RATE_VALID:
        return {
            "valid": False,
            "reason": f"cara detectada solo en {detected}/{total} frames "
                      f"({detection_rate*100:.0f}%) -- encuadre malo, repetí la toma",
        }

    sequence, mask = preprocess_sequence(filled, max_frames)
    sequence, mask = sequence.to(DEVICE), mask.to(DEVICE)
    with torch.inference_mode():
        logits = model(sequence, src_key_padding_mask=mask)
        probs = torch.softmax(logits, dim=1)[0]
    print(f"[predict] modelo Conformer listo ({time.time() - t_start:.1f}s total)")

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    top_idx = int(sorted_idx[0].item())
    top_prob = float(sorted_probs[0].item())
    second_prob = float(sorted_probs[1].item()) if len(sorted_probs) > 1 else 0.0
    margin = top_prob - second_prob
    clase_predicha = class_names[top_idx]
    palabra, pronunciacion_correcta, tipo_error = _parse_clase_predicha(clase_predicha)

    # Solo confiamos en el resultado (bien o mal) si el modelo está seguro
    # -- si no llega al umbral, se lo tratamos como "no concluyente" en vez
    # de arriesgar una devolución equivocada al chico.
    confianza_suficiente = top_prob >= MIN_CONFIDENCE_PROB and margin >= MIN_CONFIDENCE_MARGIN

    all_probs = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

    # Guardamos el clip siempre que haya un resultado confiable (correcto o
    # incorrecto) -- así el historial de práctica sirve tanto para mostrar
    # avances como para que el logopeda/padre revise los errores.
    capture_file = None
    if confianza_suficiente:
        capture_file = _save_practice_clip(clase_predicha, tmp_path, suffix)

    return {
        "valid": True,
        "palabra": palabra,
        "correcta": pronunciacion_correcta if confianza_suficiente else None,
        # Subtipo de error (ej. "lambdacismo") si la clase lo trae y la
        # confianza alcanza -- None si fue correcta, si no hay confianza
        # suficiente, o si el checkpoint no distingue subtipos para esa
        # palabra (solo tiene "<palabra>_incorrecto" genérico).
        "tipo_error": tipo_error if confianza_suficiente else None,
        # Nombre de clase completo (ej. "perro_incorrecto_lambdacismo") -- lo
        # necesita el cliente para armar la URL de GET /dataset/thumbnail?word=...
        "class_name": clase_predicha,
        "prob": top_prob,
        "detected": detected,
        "total": total,
        "confianza_suficiente": confianza_suficiente,
        # Nombre del clip guardado (o None) -- la app arma la URL de la
        # foto real con GET /dataset/thumbnail?word=...&filename=... usando
        # este valor, sin tener que esperar a recargar el historial.
        "capture_file": capture_file,
        "all_probs": all_probs,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="El servidor todavía está cargando el modelo.")

    landmarker = _state["landmarker"]
    model = _state["model"]
    class_names = _state["class_names"]
    max_frames = _state["max_frames"]

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Llamada DIRECTA y bloqueante, en el mismo hilo que el event loop
        # (el principal) -- ver el comentario grande sobre esto más arriba.
        # Si esto se cuelga, ya no hay timeout interno que lo detecte (el
        # propio hilo que lo controlaría queda bloqueado) -- lo detecta y
        # mata el proceso el vigilante EXTERNO (ver server/watchdog.sh).
        return _run_prediction_pipeline(tmp_path, suffix, landmarker, model, class_names, max_frames)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn

    # El propio proceso escribe SU PID real de Windows acá -- bash ($! en
    # server/run.sh) puede devolver un PID de espacio MSYS que no coincide
    # con el PID real que Windows/taskkill necesitan (confirmado: causó que
    # el vigilante fallara al intentar matar un proceso colgado). os.getpid()
    # desde Python corriendo nativo en Windows siempre es el PID correcto.
    #
    # OJO con la ruta: "/tmp/..." es de Git-Bash/MSYS -- el python.exe
    # NATIVO de Windows no la entiende y tira FileNotFoundError (confirmado
    # en producción, tumbaba el server en loop apenas arrancaba). Se usa
    # una ruta DENTRO del proyecto (server/.server.pid) en vez de /tmp,
    # porque esa sí resuelve bien tanto desde Python nativo como desde
    # server/watchdog.sh en Git-Bash.
    pid_file = os.path.join(BASE_DIR, "server", ".server.pid")
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    uvicorn.run(app, host="127.0.0.1", port=8000)
