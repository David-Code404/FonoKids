"""
server/main.py
---------------
API HTTP de FonoKids para práctica de pronunciación infantil. Recibe un
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
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime

import av
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# audio_pronunciation.py vive en scripts/, no es un paquete instalado --
# scripts/ no está en sys.path por default cuando este archivo se corre
# como "python server/main.py" desde la raíz del proyecto.
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from audio_pronunciation import (  # noqa: E402
    EMBEDDING_DIM,
    TARGET_SAMPLE_RATE,
    extract_embedding,
    load_audio_16k,
    load_model as load_audio_model,
    score_pronunciation,
    score_word_in_sentence,
)

from train_audio_classifier import AudioClassifierHead  # noqa: E402

import db  # noqa: E402

AUDIO_CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "audio_classifier.pth")

SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")
# Dataset viejo archivado (ver .gitignore) -- la mayoría de las palabras ya
# no tienen sus clips crudos en SESSIONS_DIR (solo quedó el embedding
# precalculado, ver data/audio_embeddings_cache/), pero "perro" sigue
# teniendo cientos de .wav reales acá -- se usan como referencia de audio
# real cuando existen (ver /reference_audio/<palabra>).
ARCHIVADO_SESSIONS_DIR = os.path.join(BASE_DIR, "dataARCHIVADO", "sesiones_continuas")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pth")  # igual que probar_modelo.py
GOAL_PER_WORD = 500

MIN_FRAMES_VALID = 15           # menos que esto = probablemente no se dijo la frase completa
MIN_DETECTION_RATE_VALID = 0.6  # menos del 60% de frames con cara detectada = toma mala

# Ancho máximo de cada frame ANTES de mandarlo a HRNet -- la cámara del
# navegador graba en Full HD (1920px), muchísimo más de lo necesario para
# detectar una cara. 640px es lo mismo que usa probar_modelo.py (FRAME_SIZE).
MAX_FRAME_WIDTH = 640

# Umbral usado SOLO por _run_audio_only_pipeline (modo "solo voz", sin
# video) -- ahí no hay señal visual para cruzar, así que decide el audio
# solo con su propio umbral de confianza.
AUDIO_MIN_CONFIDENCE = 0.60

# =====================================================================
# MATRIZ DE DECISIÓN CRUZADA Acoustic_Match x Visual_Match (pedido
# explícito del docente, ver _run_prediction_pipeline) -- cada señal tiene
# su propio umbral de confianza, evaluado por separado, y las dos tienen
# que confirmar para dar "SABE" (bien dicho). Reemplaza al esquema anterior
# de "prioridad" (MIN_CONFIDENCE_PROB/MARGIN + AUDIO_MIN_CONFIDENCE con el
# audio mandando solo) que se usaba antes en el pipeline con cámara.
# =====================================================================
ACOUSTIC_MATCH_THRESHOLD = 0.80
VISUAL_MATCH_THRESHOLD = 0.75

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
# DETECCIÓN HRNet/FAN -- copiado de scripts/extraer_landmarks_npy.py,
# esquema iBUG/300W de 68 puntos: boca = índices 48-67 (20 puntos).
# =====================================================================
LIP_INDICES = list(range(48, 68))
# Comisuras de la boca (48 = derecha, 54 = izquierda) -- referencia de
# normalización en vez de los ojos, ver normalize_lip_landmarks más abajo.
MOUTH_RIGHT_CORNER = 48
MOUTH_LEFT_CORNER = 54
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
                # Probado y descartado (ver extraer_landmarks_npy.py): forzar
                # el frame completo cuando SFD no encuentra cara no rescata
                # nada, produce puntos inventados lejos de la boca real.
                resultados.append(None)
                continue
            preds = detector.get_landmarks_from_image(frame_rgb, detected_faces=faces)
        resultados.append(preds[0].astype(np.float32) if preds else None)
    return resultados


def normalize_lip_landmarks(landmarks_px):
    """landmarks_px: array (68, 2) en píxeles (esquema HRNet). Devuelve un
    vector (40,) con los 20 puntos de labios normalizados (traslación +
    rotación + escala), o None si algo sale mal (ej. comisuras casi
    superpuestas). Normaliza contra las COMISURAS DE LA BOCA, no los ojos
    -- ver el comentario largo en extraer_landmarks_npy.py sobre por qué."""
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


app = FastAPI(title="FonoKids API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global: modelo y landmarker se cargan UNA sola vez al arrancar,
# no en cada request (cargar el modelo por request sería lentísimo).
_state = {
    "model": None, "landmarker": None, "class_names": None, "max_frames": None,
    "audio_model": None, "audio_feature_extractor": None, "audio_vocab": None,
    # Clasificador de audio ENTRENADO con clips reales (ver
    # train_audio_classifier.py) -- distinto del audio_model de wav2vec2 de
    # arriba (que es genérico, sin entrenar, usado para el GOP fonema por
    # fonema). Este predice directamente la clase (correcto/incorrecto_*)
    # a partir del embedding del clip, igual que el modelo visual hace con
    # landmarks. None si todavía no se entrenó ninguno.
    "audio_clf": None, "audio_clf_classes": None, "audio_clf_mean": None, "audio_clf_std": None,
}

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
    db.init_db()

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

    # Modelo de audio (fonemas, wav2vec2) -- opcional: si falla la carga (ej.
    # sin internet la primera vez, que baja ~1.2GB), el server sigue
    # funcionando solo con el veredicto visual, sin tirar todo abajo.
    try:
        print("Cargando modelo de audio (wav2vec2, puede tardar la primera vez)...")
        audio_model, audio_fe, audio_vocab, _ = load_audio_model(device=DEVICE)
        _state.update(audio_model=audio_model, audio_feature_extractor=audio_fe, audio_vocab=audio_vocab)
        print("Modelo de audio listo.")
    except Exception as e:
        print(f"[AVISO] No se pudo cargar el modelo de audio ({e}) -- "
              "el veredicto va a usar solo el video, sin análisis de sonido.")

    # Clasificador de audio entrenado con clips reales -- opcional (puede no
    # existir todavía si nadie corrió train_audio_classifier.py). Sin esto,
    # el análisis de audio sigue funcionando con el GOP genérico de arriba,
    # solo que sin el refuerzo directo entrenado con tus datos.
    if os.path.exists(AUDIO_CLASSIFIER_PATH):
        try:
            ckpt = safe_torch_load(AUDIO_CLASSIFIER_PATH)
            clf = AudioClassifierHead(
                ckpt["embedding_dim"], ckpt["hidden_dim"], len(ckpt["class_names"]), dropout=0.0,
            ).to(DEVICE)
            clf.load_state_dict(ckpt["state_dict"])
            clf.eval()
            _state.update(
                audio_clf=clf,
                audio_clf_classes=ckpt["class_names"],
                audio_clf_mean=ckpt["mean"],
                audio_clf_std=ckpt["std"],
            )
            print(f"Clasificador de audio entrenado cargado ({len(ckpt['class_names'])} clases, "
                  f"val_acc guardado: {ckpt.get('val_acc', '?')}).")
        except Exception as e:
            print(f"[AVISO] No se pudo cargar el clasificador de audio entrenado ({e}) -- "
                  "sigo solo con el GOP genérico.")
    else:
        print(f"[AVISO] No hay clasificador de audio entrenado todavía ({AUDIO_CLASSIFIER_PATH} "
              "no existe) -- corré train_audio_classifier.py cuando tengas datos suficientes.")


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
    del lado del cliente.

    Fuente de datos: primero intenta MySQL (ver db.py -- persiste aunque
    se pierdan o se muevan los archivos de data/sesiones_continuas/, como
    ya pasó una vez en esta PC). Si la base no está disponible, cae al
    barrido del filesystem de siempre."""
    from_db = db.get_recordings_grouped()
    if from_db is not None:
        return {"recordings": from_db}

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


# Cachea la secuencia de puntos ya calculada por palabra -- correrla de
# nuevo en cada GET sería carísimo (FAN por HRNet) para algo que no cambia
# a menos que se agreguen clips nuevos de esa palabra. Se invalida sola si
# el nombre del archivo elegido cambia (ver _pick_reference_clip).
_reference_cache = {}


def _pick_reference_clip(palabra):
    """Elige un clip de "<palabra>_correcto/clips/" para usar de modelo a
    imitar -- el MÁS LARGO de los que haya (más frames = más margen para
    que la extracción encuentre uno bien encuadrado), no simplemente el
    primero por orden alfabético."""
    clips_dir = os.path.join(SESSIONS_DIR, f"{palabra}_correcto", "clips")
    if not os.path.isdir(clips_dir):
        return None
    candidatos = [f for f in os.listdir(clips_dir) if f.lower().endswith(VIDEO_EXTENSIONS)]
    if not candidatos:
        return None

    def _n_frames(filename):
        cap = cv2.VideoCapture(os.path.join(clips_dir, filename))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    mejor = max(candidatos, key=_n_frames)
    return os.path.join(clips_dir, mejor), mejor


@app.get("/reference/{palabra}")
def reference_landmarks(palabra: str):
    """Secuencia de posición de los 20 puntos de labios (SIN velocidad ni
    aceleración -- acá solo hace falta la forma, no la dinámica derivada)
    de un clip real de "<palabra>_correcto", para animar una "boquita" de
    referencia en la app y que el chico tenga algo real que imitar antes de
    grabar. Corre el detector DIRECTO sobre el clip (no depende de
    data/landmarks_npy/, que es una carpeta de trabajo transitoria que
    puede no estar generada, o desactualizada)."""
    if _state["landmarker"] is None:
        raise HTTPException(status_code=503, detail="El servidor todavía está cargando el modelo.")

    safe_palabra = os.path.basename(palabra)
    elegido = _pick_reference_clip(safe_palabra)
    if elegido is None:
        raise HTTPException(
            status_code=404,
            detail=f"No hay clips grabados en {safe_palabra}_correcto/clips/ todavía.",
        )
    clip_path, filename = elegido

    cached = _reference_cache.get(safe_palabra)
    if cached is not None and cached["filename"] == filename:
        return cached["response"]

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=422, detail="No se pudo leer el clip de referencia.")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / w
            frame = cv2.resize(frame, (MAX_FRAME_WIDTH, round(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    landmarks_por_frame = detect_landmarks_batch(_state["landmarker"], frames)
    positions = []
    last_valid_vector = None
    for landmarks_px in landmarks_por_frame:
        vector = normalize_lip_landmarks(landmarks_px) if landmarks_px is not None else None
        if vector is not None and is_outlier(vector, last_valid_vector):
            vector = None
        if vector is not None:
            last_valid_vector = vector
        positions.append(vector)

    filled = fill_missing_frames(positions)
    if filled is None:
        raise HTTPException(status_code=422, detail="No se detectó ninguna cara en el clip de referencia.")
    smoothed = smooth_positions(filled)

    # (T, 40) -> lista de frames, cada uno lista de 20 pares [x, y] --
    # formato directo para dibujar en el canvas del cliente sin reprocesar.
    frames_out = [smoothed[t].reshape(20, 2).tolist() for t in range(smoothed.shape[0])]
    response = {"palabra": safe_palabra, "frames": frames_out}
    _reference_cache[safe_palabra] = {"filename": filename, "response": response}
    return response


# Formatos que puede traer un clip real de "<palabra>_correcto/clips/" --
# .wav (grabado por grabar_video_continuo.py) se sirve directo, los de
# video (.avi/.webm) traen el audio adentro y hay que extraerlo con PyAV.
_AUDIO_DIRECTO = (".wav",)


def _buscar_clips_de_audio(palabra):
    """Todos los clips de "<palabra>_correcto/clips/" que tengan audio
    aprovechable -- busca primero en el dataset en uso (SESSIONS_DIR) y
    también en el archivado (ARCHIVADO_SESSIONS_DIR, donde quedó la mayoría
    de los clips viejos, ver comentario de esa constante). Devuelve una
    lista de rutas completas, vacía si no hay ninguna."""
    encontrados = []
    for base_dir in (SESSIONS_DIR, ARCHIVADO_SESSIONS_DIR):
        clips_dir = os.path.join(base_dir, f"{palabra}_correcto", "clips")
        if not os.path.isdir(clips_dir):
            continue
        for f in os.listdir(clips_dir):
            if f.lower().endswith(_AUDIO_DIRECTO + VIDEO_EXTENSIONS):
                encontrados.append(os.path.join(clips_dir, f))
    return encontrados


def _extraer_audio_de_video(ruta_video):
    """Pista de audio de un .avi/.webm como tensor 1D float32 mono a 16kHz
    -- misma lógica que se usa para armar el dataset de audio a partir de
    clips de video grabados con cámara (el audio viene mezclado en el
    contenedor, no como archivo aparte)."""
    contenedor = av.open(ruta_video)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
    bloques = []
    for frame in contenedor.decode(audio=0):
        for frame_resampleado in resampler.resample(frame):
            bloques.append(frame_resampleado.to_ndarray())
    contenedor.close()
    audio = np.concatenate(bloques, axis=1).flatten().astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


@app.get("/reference_audio/{palabra}")
def reference_audio(palabra: str):
    """Un clip de AUDIO real (no sintetizado) de alguien diciendo
    "<palabra>" correctamente -- pedido explícito: mejor que el chico
    escuche cómo suena de verdad dicho bien, no solo una voz de
    texto-a-voz genérica, cuando hay un clip real disponible para esa
    palabra. La mayoría de las palabras hoy NO tienen clips crudos
    guardados (se archivaron para no pesar el repo, ver
    ARCHIVADO_SESSIONS_DIR) -- el frontend tiene que caer a TTS si esto
    devuelve 404, no todas las palabras van a tener audio real."""
    safe_palabra = os.path.basename(palabra)
    candidatos = _buscar_clips_de_audio(safe_palabra)
    if not candidatos:
        raise HTTPException(
            status_code=404,
            detail=f"No hay clips de audio real guardados para '{safe_palabra}'.",
        )

    elegido = random.choice(candidatos)
    try:
        if elegido.lower().endswith(_AUDIO_DIRECTO):
            waveform = load_audio_16k(elegido)
        else:
            waveform = _extraer_audio_de_video(elegido)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No se pudo leer el clip de referencia ({e}).")

    buffer_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    buffer_wav.close()
    try:
        torchaudio.save(buffer_wav.name, waveform.unsqueeze(0), TARGET_SAMPLE_RATE)
        with open(buffer_wav.name, "rb") as f:
            contenido = f.read()
    finally:
        os.remove(buffer_wav.name)

    return Response(content=contenido, media_type="audio/wav")


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


def _score_audio_pronunciation(audio_path, palabra):
    """Envoltorio fino sobre audio_pronunciation.score_pronunciation() --
    nunca tira: si el audio no se pudo leer/puntuar, devuelve None y el
    veredicto final cae solo en el video (ver _run_prediction_pipeline)."""
    if audio_path is None or not palabra:
        return None
    if _state["audio_model"] is None:
        return None
    try:
        return score_pronunciation(
            audio_path, palabra,
            _state["audio_model"], _state["audio_feature_extractor"], _state["audio_vocab"], DEVICE,
        )
    except Exception as e:
        print(f"[AVISO] Falló el análisis de audio ({e}) -- sigo solo con video.")
        return None


def _score_word_in_sentence(audio_path, frase, palabra):
    """Envoltorio fino sobre audio_pronunciation.score_word_in_sentence() --
    nunca tira: si falla, devuelve None."""
    if audio_path is None or not frase or not palabra:
        return None
    if _state["audio_model"] is None:
        return None
    try:
        return score_word_in_sentence(
            audio_path, frase, palabra,
            _state["audio_model"], _state["audio_feature_extractor"], _state["audio_vocab"], DEVICE,
        )
    except Exception as e:
        print(f"[AVISO] Falló el análisis de frase ({e}).")
        return None


def _classify_audio(audio_path, palabra):
    """Clasificador de audio ENTRENADO con clips reales (ver
    train_audio_classifier.py) -- a diferencia del GOP de arriba (que fuerza
    el audio contra la palabra esperada pase lo que pase), esto predice
    directamente entre las clases que existan para esa palabra
    ("<palabra>_correcto", "<palabra>_incorrecto_*"), así que SÍ puede
    reconocer que el audio no se parece a nada de lo esperado.

    Devuelve {"clase_predicha", "prob", "es_correcto"} o None si no hay
    clasificador entrenado, o si esa palabra no tiene clases entrenadas
    todavía (el clasificador solo conoce las palabras con clips grabados)."""
    clf = _state["audio_clf"]
    if clf is None or audio_path is None or not palabra:
        return None
    class_names = _state["audio_clf_classes"]
    # Solo tiene sentido clasificar si HAY clases de esta palabra entre las
    # que el clasificador aprendió -- si pedís "perro" pero solo entrenaste
    # con "carro", no hay nada que comparar.
    #
    # BUG real detectado (caso "perro"): un simple startswith(f"{palabra}_")
    # también agarraba las sub-familias de RR ("perro_medias_correcto",
    # "perro_doble_correcto") cuando se pedía la palabra COMPLETA ("perro"),
    # porque esos nombres también empiezan con "perro_". Eso contaminaba la
    # familia: decir "perro" mal podía terminar comparado contra clips de
    # "perro a medias" (un pedacito, acústicamente muy distinto), no contra
    # un "perro" completo mal dicho. Ahora matchea SOLO "<palabra>_correcto"
    # o "<palabra>_incorrecto*" -- las sub-familias piden su propio prefijo
    # exacto (ej. "perro_medias"), así que no se cuelan acá.
    indices_de_la_palabra = [
        i for i, c in enumerate(class_names)
        if c == f"{palabra}_correcto" or c.startswith(f"{palabra}_incorrecto")
    ]
    if not indices_de_la_palabra:
        return None
    try:
        emb = extract_embedding(audio_path, _state["audio_model"], _state["audio_feature_extractor"], DEVICE)
        emb_norm = (emb - _state["audio_clf_mean"]) / _state["audio_clf_std"]
        # mean/std se guardaron con shape (1, 1024) (ver train_audio_
        # classifier.py) -- emb_norm ya sale (1, 1024) por el broadcasting,
        # así que reshape en vez de unsqueeze (que agregaba una dimensión de
        # más y rompía el softmax/indexado de abajo).
        x = torch.from_numpy(emb_norm.reshape(1, -1)).float().to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(clf(x), dim=1)[0]
        # BUG real detectado: acá se calculaba el argmax sobre las 58 clases
        # completas, sin usar "indices_de_la_palabra" para nada -- si pedías
        # "perro" pero el audio se parecía más a "gota", devolvía
        # "gota_correcto" en vez de reconocer que ni siquiera es la palabra
        # pedida. Ahora el argmax se restringe a los índices de ESA palabra
        # (correcto/incorrecto/medias/doble, lo que tenga entrenado) -- si el
        # audio no se parece a nada de esa familia, la probabilidad ahí
        # dentro sale baja de por sí (la masa está en otras clases), así que
        # cae solo bajo AUDIO_MIN_CONFIDENCE sin necesitar lógica aparte.
        probs_familia = probs[indices_de_la_palabra]
        idx_local = int(torch.argmax(probs_familia).item())
        idx = indices_de_la_palabra[idx_local]
        clase_predicha = class_names[idx]
        return {
            "clase_predicha": clase_predicha,
            "prob": float(probs[idx].item()),
            "es_correcto": clase_predicha == f"{palabra}_correcto",
        }
    except Exception as e:
        print(f"[AVISO] Falló el clasificador de audio entrenado ({e}) -- sigo solo con el GOP.")
        return None


def _run_audio_only_pipeline(audio_path, palabra_esperada):
    """Versión SIN VIDEO de _run_prediction_pipeline -- para cuando el chico
    elige practicar sin cámara (ver CameraConsentGate en el frontend, modo
    "solo con la voz"). Usa EXCLUSIVAMENTE las dos señales de audio (GOP +
    clasificador entrenado) para decidir el veredicto, mismo umbral de
    confianza (AUDIO_MIN_CONFIDENCE) que ya usa el pipeline con video para
    dejar que el audio decida solo. Sin video de respaldo: si el audio no
    está seguro, el resultado queda "no concluyente" directamente."""
    t_start = time.time()

    es_palabra_completa = palabra_esperada and "_" not in palabra_esperada
    audio_resultado = (
        _score_audio_pronunciation(audio_path, palabra_esperada) if es_palabra_completa else None
    )
    audio_ok = (
        audio_resultado is not None
        and len(audio_resultado["sospechosos"]) == 0
        and audio_resultado["palabra_reconocida"]
    )
    audio_clf_resultado = _classify_audio(audio_path, palabra_esperada)
    prob_str = f"{audio_clf_resultado['prob']:.2f}" if audio_clf_resultado else "?"
    print(f"[predict-audio] audio GOP {'OK' if audio_resultado is None else ('bien' if audio_ok else 'sospechoso')} | "
          f"clasificador {'sin entrenar' if audio_clf_resultado is None else audio_clf_resultado['clase_predicha']} "
          f"prob={prob_str} (umbral {AUDIO_MIN_CONFIDENCE}) "
          f"({time.time() - t_start:.1f}s total)")

    if audio_clf_resultado is None:
        return {
            "valid": False,
            "reason": "no se pudo analizar el audio -- probá de nuevo",
        }

    audio_confiable = audio_clf_resultado["prob"] >= AUDIO_MIN_CONFIDENCE
    # Mismo fix que en _run_prediction_pipeline: cuando el audio está
    # seguro, decide el clasificador ENTRENADO solo, sin que el GOP genérico
    # (más propenso a falsos "sospechosos" en palabras largas como
    # "serpiente") lo tape.
    audio_dice_bien = audio_clf_resultado["es_correcto"]

    clase_predicha = audio_clf_resultado["clase_predicha"]
    palabra, _pronunciacion_correcta, tipo_error = _parse_clase_predicha(clase_predicha)

    if not audio_confiable:
        veredicto_final = "no_concluyente"
    else:
        veredicto_final = "bien" if audio_dice_bien else "a_practicar"

    # El dataset real hoy no separa clases por subtipo (todo queda mezclado
    # en "_incorrecto"), así que tipo_error de la clase casi siempre es None
    # -- en vez de resignarse, se lo deriva del audio (GOP + reconocimiento
    # libre, ver audio_pronunciation.diagnose_tipo_error): compara lo que se
    # esperaba contra lo que realmente se escuchó en la posición del sonido
    # difícil, sin necesitar clases entrenadas por subtipo.
    if tipo_error is None and veredicto_final == "a_practicar" and audio_resultado is not None:
        tipo_error = audio_resultado.get("tipo_error_audio")

    confianza_suficiente = audio_confiable

    if confianza_suficiente:
        db.save_attempt(
            class_name=clase_predicha, palabra=palabra, correcta=(veredicto_final == "bien"),
            tipo_error=tipo_error if veredicto_final == "a_practicar" else None,
            prob=audio_clf_resultado["prob"],
        )

    return {
        "valid": True,
        "palabra": palabra,
        "correcta": (veredicto_final == "bien") if confianza_suficiente else None,
        "veredicto_final": veredicto_final,
        "audio": {
            "fonemas": audio_resultado["fonemas"],
            "scores": audio_resultado["scores"],
            "sospechosos": audio_resultado["sospechosos"],
            "reconocido_libre": audio_resultado["reconocido_libre"],
            "sonido_reconocido": audio_resultado["sonido_reconocido"],
            "palabra_reconocida": audio_resultado["palabra_reconocida"],
        } if audio_resultado is not None else None,
        "audio_clasificador": audio_clf_resultado,
        "tipo_error": tipo_error if (confianza_suficiente and veredicto_final == "a_practicar") else None,
        # Modo solo-audio: el clasificador solo compara contra las clases de
        # la palabra pedida (ver _classify_audio), nunca puede detectar que
        # se dijo OTRA palabra -- siempre False acá.
        "palabra_distinta": False,
        "class_name": clase_predicha,
        "prob": audio_clf_resultado["prob"],
        "detected": None,
        "total": None,
        "confianza_suficiente": confianza_suficiente,
        "capture_file": None,
        "all_probs": None,
    }


def _run_prediction_pipeline(tmp_path, suffix, landmarker, model, class_names, max_frames,
                              audio_path=None, palabra_esperada=None):
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
    print(f"[predict] video: clase={clase_predicha} prob={top_prob:.2f} margin={margin:.2f} "
          f"(2da opcion: {class_names[int(sorted_idx[1].item())] if len(sorted_idx) > 1 else '?'} "
          f"{second_prob:.2f}) | esperado={palabra_esperada}")
    palabra, pronunciacion_correcta, tipo_error = _parse_clase_predicha(clase_predicha)

    # Cada paso del camino (sonido suelto / doble / a medias / completa) es
    # una FAMILIA de clases distinta ("perro", "perro_doble",
    # "perro_medias", ...) -- si el chico está practicando "perro_medias"
    # (decir "per") pero el modelo reconoce algo de OTRA familia (ej.
    # "perro_correcto", la palabra completa), eso está MAL para este paso
    # puntual aunque técnicamente haya sonado bien esa otra cosa -- por
    # algo el camino separa los pasos, no sirve de nada si decir la
    # palabra entera "aprueba" el paso del sonido suelto. Sin este chequeo,
    # el veredicto de abajo solo miraba si la clase ganadora terminaba en
    # "_correcto", sin importar de qué familia era.
    #
    # EXCEPCIÓN necesaria: si el paso pedido es la palabra COMPLETA (ej.
    # "perro", sin sufijo de paso) y el modelo ganador es de la MISMA
    # palabra pero dudó de paso (ej. "perro_medias_correcto" en vez de
    # "perro_correcto") -- confirmado en la matriz de confusión, es justo
    # donde más se confunde el modelo visual, porque decir "perro" entero
    # y decir "per"/"rr" sueltos se ve muy parecido en los labios -- no
    # tiene sentido penalizar la IDENTIDAD de la palabra ahí, ya sabemos
    # que es la palabra correcta. El chequeo estricto de arriba solo debe
    # aplicar cuando el paso pedido es uno PARCIAL (ahí sí importa que no
    # se cuele la palabra entera u otro paso distinto).
    def _palabra_base(nombre):
        return nombre.split("_")[0]

    # Marca aparte para "dijo OTRA palabra" (ej. pidieron "carro" y el video
    # reconoció "perro") -- distinto de "dijo la palabra pedida pero mal
    # pronunciada". El clasificador de audio NUNCA puede detectar esto (ver
    # _classify_audio: solo compara contra las clases de la palabra pedida,
    # no puede reconocer que el audio ni siquiera corresponde a esa
    # familia), así que esta señal sale únicamente del video -- pedido
    # explícito: el chico tiene que ver "esa palabra no va acá" en vez de
    # un genérico "mal dicho" cuando ni siquiera dijo la palabra que tocaba.
    palabra_distinta = False
    if palabra_esperada is not None and palabra != palabra_esperada:
        pidio_palabra_completa = palabra_esperada == _palabra_base(palabra_esperada)
        misma_palabra_base = _palabra_base(palabra) == _palabra_base(palabra_esperada)
        if not (pidio_palabra_completa and misma_palabra_base):
            pronunciacion_correcta = False
            tipo_error = None
            palabra_distinta = True

    all_probs = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

    # Análisis de AUDIO (sonido, no video) -- ver audio_pronunciation.py. Se
    # compara contra la palabra que el chico tenía que decir (la que eligió
    # en la app), no la que "cree" el modelo visual, para que sea una
    # segunda señal independiente de verdad, no un eco de la primera.
    #
    # OJO: el GOP convierte la palabra a fonemas de verdad (texto_a_fonemas),
    # así que solo tiene sentido para una palabra real completa -- las
    # familias de paso parcial (ej. "perro_medias", "perro_doble") no son
    # texto pronunciable, son identificadores de clase. Para esos pasos el
    # GOP se salta y la decisión queda en manos del video + el clasificador
    # de audio entrenado (que sí filtran por familia correctamente, ver
    # _classify_audio).
    es_palabra_completa = (palabra_esperada or palabra) and "_" not in (palabra_esperada or palabra)
    audio_resultado = (
        _score_audio_pronunciation(audio_path, palabra_esperada or palabra)
        if es_palabra_completa else None
    )
    # "palabra_reconocida" (comparación contra el reconocimiento LIBRE, sin
    # forzar nada) hace falta ADEMÁS de "sospechosos" (que sale de forced_align,
    # que siempre encaja el audio contra la palabra esperada pase lo que
    # pase) -- confirmado con un caso real: dijo "pedo" en vez de "perro" y
    # forced_align igual armó una alineación sin marcar fonemas sospechosos.
    audio_ok = (
        audio_resultado is not None
        and len(audio_resultado["sospechosos"]) == 0
        and audio_resultado["palabra_reconocida"]
    )

    # Clasificador de audio ENTRENADO con tus clips reales -- señal más
    # fuerte que el GOP (que solo fuerza fonema por fonema): aprendió
    # directo de ejemplos reales de cada clase. Si existe (ver
    # train_audio_classifier.py), pesa más que el GOP en la decisión final.
    audio_clf_resultado = _classify_audio(audio_path, palabra_esperada or palabra)
    prob_str = f"{audio_clf_resultado['prob']:.2f}" if audio_clf_resultado else "?"
    print(f"[predict] audio GOP {'OK' if audio_resultado is None else ('bien' if audio_ok else 'sospechoso')} | "
          f"clasificador {'sin entrenar' if audio_clf_resultado is None else audio_clf_resultado['clase_predicha']} "
          f"prob={prob_str} (umbral {AUDIO_MIN_CONFIDENCE}) video_prob={top_prob:.2f} "
          f"({time.time() - t_start:.1f}s total)")

    # NOTA histórica: acá antes había un esquema de "prioridad" (el audio
    # mandaba solo si estaba seguro, el video era respaldo) -- funcionaba
    # bien pero el docente pidió específicamente la matriz cruzada de abajo,
    # que evalúa las dos señales por separado y exige que confirmen las dos.
    # =====================================================================
    # MATRIZ DE DECISIÓN CRUZADA (pedido explícito del docente) -- reemplaza
    # la lógica de "prioridad" de arriba por una matriz Acoustic_Match x
    # Visual_Match, cada señal con su propio umbral, evaluadas por separado:
    #   Acoustic_Match = clasificador de audio entrenado predice "correcto"
    #                     Y su confianza >= ACOUSTIC_MATCH_THRESHOLD (0.80)
    #   Visual_Match   = el video predice la familia correcta (mismo chequeo
    #                     de familia que ya se hacía arriba) Y su confianza
    #                     (top_prob del Conformer) >= VISUAL_MATCH_THRESHOLD
    #                     (0.75)
    # Con las dos señales evaluadas, se cruzan en 4 combinaciones -- a
    # diferencia del esquema de prioridad anterior, ACÁ SÍ un video que no
    # llega a su umbral puede bajar un resultado aunque el audio esté bien
    # (eso es justamente lo que pide esta matriz: las dos señales tienen
    # que confirmar, no alcanza con que una sola esté segura).
    acoustic_match = (
        audio_clf_resultado is not None
        and audio_clf_resultado["prob"] >= ACOUSTIC_MATCH_THRESHOLD
        and audio_clf_resultado["es_correcto"]
    )
    visual_match = top_prob >= VISUAL_MATCH_THRESHOLD and pronunciacion_correcta

    # Sin clasificador de audio entrenado para esta palabra no hay señal
    # acústica para cruzar -- no se puede aplicar la matriz, resultado
    # "no_concluyente" en vez de inventar un veredicto con una sola señal.
    confianza_suficiente = audio_clf_resultado is not None

    if not confianza_suficiente:
        veredicto_final = "no_concluyente"
        tipo_error = None
    elif acoustic_match and visual_match:
        veredicto_final = "bien"
        tipo_error = None
    elif (not acoustic_match) and visual_match:
        veredicto_final = "a_practicar"
        tipo_error = "sustitucion_acustica"  # Error de Sustitución / Distorsión Acústica
    elif (not acoustic_match) and (not visual_match):
        veredicto_final = "a_practicar"
        tipo_error = "motor_severo"  # Error Motor-Articulatorio Severo
    else:  # acoustic_match and not visual_match
        veredicto_final = "a_practicar"
        tipo_error = "compensacion_visual"  # Compensación Visual / Gesto Atípico

    # Se guarda el clip siempre que haya confianza suficiente (correcto O
    # incorrecto) -- Logros solo cuenta los "bien dichos", pero la pantalla
    # de Errores necesita los intentos fallidos para poder mostrarlos.
    capture_file = None
    if confianza_suficiente:
        capture_file = _save_practice_clip(clase_predicha, tmp_path, suffix)
        db.save_attempt(
            class_name=clase_predicha, palabra=palabra, correcta=(veredicto_final == "bien"),
            tipo_error=tipo_error if veredicto_final == "a_practicar" else None, prob=top_prob,
        )

    return {
        "valid": True,
        "palabra": palabra,
        "correcta": (veredicto_final == "bien") if confianza_suficiente else None,
        "veredicto_final": veredicto_final,
        "audio": {
            "fonemas": audio_resultado["fonemas"],
            "scores": audio_resultado["scores"],
            "sospechosos": audio_resultado["sospechosos"],
            "reconocido_libre": audio_resultado["reconocido_libre"],
            "sonido_reconocido": audio_resultado["sonido_reconocido"],
            "palabra_reconocida": audio_resultado["palabra_reconocida"],
        } if audio_resultado is not None else None,
        "audio_clasificador": audio_clf_resultado,
        # Subtipo de error (ej. "lambdacismo") si la clase lo trae y la
        # confianza alcanza -- None si fue correcta, si no hay confianza
        # suficiente, o si el checkpoint no distingue subtipos para esa
        # palabra (solo tiene "<palabra>_incorrecto" genérico).
        "tipo_error": tipo_error if (confianza_suficiente and veredicto_final == "a_practicar") else None,
        # True si el video reconoció una palabra distinta a la pedida (ver
        # arriba) -- el frontend usa esto para mostrar "esa palabra no es de
        # este paso" en vez del mensaje de pronunciación.
        "palabra_distinta": palabra_distinta and veredicto_final == "a_practicar",
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
async def predict(
    file: UploadFile = File(None),
    audio: UploadFile = File(None),
    palabra: str = Form(None),
):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="El servidor todavía está cargando el modelo.")

    # El audio es OPCIONAL cuando HAY video (compatibilidad con clientes
    # viejos, y para no romper /predict si el navegador del chico no tiene
    # micrófono). Cuando NO hay video (modo "solo voz", ver
    # CameraConsentGate en el frontend), el audio pasa a ser obligatorio --
    # sin ninguna de las dos señales no hay nada que evaluar.
    audio_path = None
    if audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            shutil.copyfileobj(audio.file, tmp_audio)
            audio_path = tmp_audio.name

    if file is None:
        if audio_path is None:
            raise HTTPException(status_code=400, detail="Hace falta mandar audio o video.")
        try:
            return _run_audio_only_pipeline(audio_path, palabra_esperada=palabra)
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

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
        return _run_prediction_pipeline(
            tmp_path, suffix, landmarker, model, class_names, max_frames,
            audio_path=audio_path, palabra_esperada=palabra,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        if audio_path is not None:
            try:
                os.remove(audio_path)
            except OSError:
                pass


@app.post("/predict_frase")
async def predict_frase(
    audio: UploadFile = File(...),
    frase: str = Form(...),
    palabra: str = Form(...),
):
    """Evalúa UNA palabra ya entrenada dentro de una FRASE completa (ej.
    frase="el perro corre por las montañas", palabra="perro") -- pedido
    explícito: practicar en un contexto más natural, no solo la palabra
    sola, Y que de verdad se use el clasificador entrenado (99% accuracy
    real) en vez de depender solo del GOP genérico (~80%).

    Cómo se combina: audio_pronunciation.score_word_in_sentence() alinea
    la FRASE ENTERA contra el audio, ubica en qué tramo de tiempo cae la
    palabra objetivo, y devuelve ESE pedacito de audio ya recortado
    (audio_recortado_path) -- recién ahí se le pasa al clasificador
    entrenado (_classify_audio), exactamente como si fuera un clip de la
    palabra sola. Misma prioridad que el resto del proyecto: el clasificador
    manda cuando supera AUDIO_MIN_CONFIDENCE (señal de que el audio
    recortado sí se parece a algo que entrenó), el GOP decide cuando el
    clasificador no está seguro (audio recortado raro/fuera de lo
    entrenado) o directamente no pudo correr."""
    if _state["audio_model"] is None:
        raise HTTPException(status_code=503, detail="El servidor todavía está cargando el modelo de audio.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        shutil.copyfileobj(audio.file, tmp_audio)
        audio_path = tmp_audio.name

    audio_recortado_path = None
    try:
        resultado = _score_word_in_sentence(audio_path, frase, palabra)
        if resultado is None:
            return {"valid": False, "reason": "no se pudo analizar el audio -- probá de nuevo"}

        audio_recortado_path = resultado.get("audio_recortado_path")

        # Pedido explícito: si no dijo la FRASE completa (ej. solo la
        # palabra suelta), el intento no cuenta -- ni se guarda en la base
        # ni se evalúa la palabra, directo se le pide repetir la frase
        # entera. Ver FRASE_COMPLETA_LARGO_MINIMO en audio_pronunciation.py.
        # (audio_recortado_path ya quedó asignado arriba para que el
        # finally de abajo igual borre el temporal aunque cortemos acá.)
        if not resultado.get("frase_completa_dicha", True):
            return {
                "valid": False,
                "frase_incompleta": True,
                "reason": "decí toda la frase, despacito -- ¡vos podés!",
            }

        # Pedido explícito: si cambió otra palabra de la frase (ej. "la
        # torre es muy BAJA" en vez de "ALTA"), tampoco cuenta -- aunque la
        # palabra objetivo ("torre") haya salido bien. Ver
        # resto_de_la_frase_ok en audio_pronunciation.py.
        if not resultado.get("resto_de_la_frase_ok", True):
            return {
                "valid": False,
                "frase_distinta": True,
                "reason": "repetí la frase de arriba tal cual -- ¡dale, de nuevo!",
            }

        clf_resultado = _classify_audio(audio_recortado_path, palabra) if audio_recortado_path else None

        gop_ok = len(resultado["sospechosos"]) == 0 and resultado["palabra_reconocida"]

        # BUG real detectado (caso "gorra" -> "goda"): sacar el umbral de
        # confianza del todo (commit anterior, por el caso "sherpiente") fue
        # un arreglo mal calibrado -- "sherpiente" en realidad tenía 80% de
        # confianza real (muy por encima del umbral), así que nunca hacía
        # falta sacarlo. Sin el umbral, el clasificador también "decidía"
        # con 0-2% de confianza (el audio recortado no se parece a NADA
        # entrenado -- típico cuando el recorte de la frase sale raro) y
        # igual contaba "bien" solo porque esa clase ganó por aunque sea un
        # pelo entre las dos opciones de la familia. Se restaura el mismo
        # umbral (AUDIO_MIN_CONFIDENCE) que usa el resto del proyecto: el
        # clasificador manda cuando está genuinamente seguro, el GOP decide
        # cuando no.
        clf_confiable = clf_resultado is not None and clf_resultado["prob"] >= AUDIO_MIN_CONFIDENCE
        if clf_confiable:
            veredicto_final = "bien" if clf_resultado["es_correcto"] else "a_practicar"
            fuente = "clasificador"
        else:
            veredicto_final = "bien" if gop_ok else "a_practicar"
            fuente = "gop"
        tipo_error = resultado.get("tipo_error_audio") if veredicto_final == "a_practicar" else None

        prob_str = f"{clf_resultado['prob']:.2f}" if clf_resultado else "?"
        clase_str = clf_resultado["clase_predicha"] if clf_resultado else "sin dato"
        print(f"[predict-frase] palabra={palabra!r} en frase={frase!r} | "
              f"GOP {'bien' if gop_ok else 'sospechoso'} | "
              f"clasificador {clase_str} prob={prob_str} | decide={fuente} -> {veredicto_final}")

        db.save_attempt(
            class_name=f"{palabra}_en_frase", palabra=palabra, correcta=(veredicto_final == "bien"),
            tipo_error=tipo_error, prob=clf_resultado["prob"] if clf_resultado else (0.0 if gop_ok else 1.0),
        )

        return {
            "valid": True,
            "palabra": palabra,
            "frase": frase,
            "correcta": veredicto_final == "bien",
            "veredicto_final": veredicto_final,
            "decidido_por": fuente,
            "audio": {
                "fonemas": resultado["fonemas"],
                "scores": resultado["scores"],
                "sospechosos": resultado["sospechosos"],
                "reconocido_libre": resultado["reconocido_libre"],
                "sonido_reconocido": resultado["sonido_reconocido"],
                "palabra_reconocida": resultado["palabra_reconocida"],
            },
            "audio_clasificador": clf_resultado,
            "tipo_error": tipo_error,
            "confianza_suficiente": True,
        }
    finally:
        if audio_recortado_path:
            try:
                os.remove(audio_recortado_path)
            except OSError:
                pass
        try:
            os.remove(audio_path)
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
