# Este import va PRIMERO a propósito: si pyarrow (que transformers/pandas cargan más tarde, al abrir
# el modelo de audio) se importa después de torch/cv2/CUDA, en esta PC Windows el proceso muere con
# "heap corruption" (0xC0000374) justo al cargar wav2vec2. Cargándolo antes, el servidor arranca bien.
try:
    import pyarrow.compute  # noqa: F401
except Exception:
    pass

import math
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from audio_pronunciation import (
    TARGET_SAMPLE_RATE,
    extract_embedding,
    load_audio_16k,
    load_model as load_audio_model,
    score_pronunciation,
    score_word_in_sentence,
)

from train_audio_classifier import AudioClassifierHead

import db

AUDIO_CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "audio_classifier.pth")

SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")
ARCHIVADO_SESSIONS_DIR = os.path.join(BASE_DIR, "dataARCHIVADO", "sesiones_continuas")
# [LABIOS DESACTIVADO] MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pth")
MODELO_CARA_PATH = os.path.join(BASE_DIR, "data", "modelos_cara_completa", "best_cara_completa.pth")
# [LABIOS DESACTIVADO] USAR_MODELO_CARA = os.environ.get("FONOKIDS_CARA", "1") != "0"
MIN_DETECCION_CARA = 0.6
OJO_DERECHO_EXTERNO = 36
OJO_IZQUIERDO_EXTERNO = 45
GOAL_PER_WORD = 500

MIN_FRAMES_VALID = 15
MIN_DETECTION_RATE_VALID = 0.6

MAX_FRAME_WIDTH = 640

AUDIO_MIN_CONFIDENCE = 0.60

AUDIO_CONFIRMA_CON_GOP = 0.30
ACOUSTIC_MATCH_THRESHOLD = 0.80
VISUAL_MATCH_THRESHOLD = 0.75

SUFIJO_CORRECTO = "_correcto"
MARCADOR_INCORRECTO = "_incorrecto"


def _parse_clase_predicha(nombre_clase):
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

LIP_INDICES = list(range(48, 68))
MOUTH_RIGHT_CORNER = 48
MOUTH_LEFT_CORNER = 54
SCALE_EPSILON = 1e-6
HRNET_BATCH_CHUNK_SIZE = 4


def build_hrnet_detector():
    import face_alignment
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    if DEVICE == "cpu":
        print("[AVISO] No hay GPU disponible -- HRNet/FAN en CPU va a ser MUY lento.")

    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=DEVICE, flip_input=False, compile=False,
    )
    print(f">>> Detector HRNet/FAN corriendo en {DEVICE.upper()}. <<<")
    return detector


def detect_landmarks_batch(detector, frames_bgr):
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


def normalize_face_landmarks(landmarks_px):
    ojo_derecho = landmarks_px[OJO_DERECHO_EXTERNO].astype(np.float64)
    ojo_izquierdo = landmarks_px[OJO_IZQUIERDO_EXTERNO].astype(np.float64)
    centro = (ojo_derecho + ojo_izquierdo) / 2.0
    vector_ojos = ojo_izquierdo - ojo_derecho
    escala = np.linalg.norm(vector_ojos)
    if escala < 1e-3:
        return None
    angulo = math.atan2(vector_ojos[1], vector_ojos[0])
    cos_a, sin_a = math.cos(-angulo), math.sin(-angulo)
    rotacion = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    puntos = landmarks_px[:68].astype(np.float64)
    normalizados = ((puntos - centro) @ rotacion.T) / (escala + SCALE_EPSILON)
    return normalizados.astype(np.float32).flatten()


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

_state = {
    "model": None, "landmarker": None, "class_names": None, "max_frames": None,
    "audio_model": None, "audio_feature_extractor": None, "audio_vocab": None,
    "audio_clf": None, "audio_clf_classes": None, "audio_clf_mean": None, "audio_clf_std": None,
    "model_cara": None, "class_names_cara": None, "max_frames_cara": None,
}


@app.on_event("startup")
async def load_everything():
    db.init_db()

    if not os.path.exists(MODELO_CARA_PATH):
        print(f"[AVISO] No encontré {MODELO_CARA_PATH} -- /predict queda deshabilitado. "
              "Entrenalo con Colab/Entrenar_Cara_Completa_FonoKids.ipynb. "
              "El resto del server (dashboard, stats) funciona igual.")
        return

    ckpt_cara = safe_torch_load(MODELO_CARA_PATH)
    class_names = ckpt_cara["class_names"]
    max_frames = ckpt_cara["max_frames"]
    modelo_cara = LipReadingConformer(num_classes=len(class_names), input_dim=ckpt_cara["input_dim"]).to(DEVICE)
    modelo_cara.load_state_dict(ckpt_cara["model_state_dict"])
    modelo_cara.eval()
    print(f"Modelo de cara completa cargado ({len(class_names)} clases, entrada {ckpt_cara['input_dim']}, "
          f"val_acc guardado: {ckpt_cara.get('val_acc', '?')}).")

    # [LABIOS DESACTIVADO] if not os.path.exists(MODEL_PATH):
        # [LABIOS DESACTIVADO] print(f"[AVISO] No encontré {MODEL_PATH} -- /predict queda deshabilitado. "
              # [LABIOS DESACTIVADO] "Entrenalo con train_landmarks_transformer.py o bajalo de Colab. "
              # [LABIOS DESACTIVADO] "El resto del server (dashboard, stats) funciona igual.")
        # [LABIOS DESACTIVADO] return

    # [LABIOS DESACTIVADO] checkpoint = safe_torch_load(MODEL_PATH)
    # [LABIOS DESACTIVADO] class_names = checkpoint["class_names"]
    # [LABIOS DESACTIVADO] max_frames = checkpoint["max_frames"]
    # [LABIOS DESACTIVADO] input_dim = checkpoint.get("input_dim", 120)

    # [LABIOS DESACTIVADO] model = LipReadingConformer(num_classes=len(class_names), input_dim=input_dim).to(DEVICE)
    # [LABIOS DESACTIVADO] model.load_state_dict(checkpoint["model_state_dict"])
    # [LABIOS DESACTIVADO] model.eval()

    print("Cargando detector de landmarks (HRNet/FAN)...")
    landmarker = build_hrnet_detector()

    if DEVICE == "cuda":
        try:
            print("Precalentando HRNet (primera compilación, puede tardar)...")
            dummy_frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(3)]
            detect_landmarks_batch(landmarker, dummy_frames)
            print("Precalentamiento listo.")
        except Exception as e:
            print(f"[AVISO] Precalentamiento falló ({e}), no es grave -- "
                  "la primera predicción real puede tardar más de lo normal.")

    # [LABIOS DESACTIVADO] _state.update(model=model, landmarker=landmarker, class_names=class_names, max_frames=max_frames)
    _state.update(landmarker=landmarker, model_cara=modelo_cara, class_names_cara=class_names, max_frames_cara=max_frames)

    # [LABIOS DESACTIVADO] if USAR_MODELO_CARA and os.path.exists(MODELO_CARA_PATH):
        # [LABIOS DESACTIVADO] try:
            # [LABIOS DESACTIVADO] ckpt_cara = safe_torch_load(MODELO_CARA_PATH)
            # [LABIOS DESACTIVADO] modelo_cara = LipReadingConformer(
                # [LABIOS DESACTIVADO] num_classes=len(ckpt_cara["class_names"]), input_dim=ckpt_cara["input_dim"],
            # [LABIOS DESACTIVADO] ).to(DEVICE)
            # [LABIOS DESACTIVADO] modelo_cara.load_state_dict(ckpt_cara["model_state_dict"])
            # [LABIOS DESACTIVADO] modelo_cara.eval()
            # [LABIOS DESACTIVADO] _state.update(model_cara=modelo_cara, class_names_cara=ckpt_cara["class_names"],
                          # [LABIOS DESACTIVADO] max_frames_cara=ckpt_cara["max_frames"])
            # [LABIOS DESACTIVADO] print(f"Modelo de cara completa cargado ({len(ckpt_cara['class_names'])} clases, "
                  # [LABIOS DESACTIVADO] f"entrada {ckpt_cara['input_dim']}, val_acc guardado: {ckpt_cara.get('val_acc', '?')}).")
        # [LABIOS DESACTIVADO] except Exception as e:
            # [LABIOS DESACTIVADO] print(f"[AVISO] No se pudo cargar el modelo de cara completa ({e}) -- sigo solo con el de labios.")
    # [LABIOS DESACTIVADO] else:
        # [LABIOS DESACTIVADO] print("Modelo de cara completa desactivado o no encontrado -- se usa solo el de labios.")
    # [LABIOS DESACTIVADO] print(f"Listo. Device: {DEVICE}. Clases ({len(class_names)}): {class_names}")
    print(f"Listo. Device: {DEVICE}. Solo cara completa. Clases ({len(class_names)}): {class_names}")

    try:
        print("Cargando modelo de audio (wav2vec2, puede tardar la primera vez)...")
        audio_model, audio_fe, audio_vocab, _ = load_audio_model(device=DEVICE)
        _state.update(audio_model=audio_model, audio_feature_extractor=audio_fe, audio_vocab=audio_vocab)
        print("Modelo de audio listo.")
    except Exception as e:
        print(f"[AVISO] No se pudo cargar el modelo de audio ({e}) -- "
              "el veredicto va a usar solo el video, sin análisis de sonido.")

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
    return sequence.unsqueeze(0), mask


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "classes": _state["class_names_cara"],
            "cara_completa": _state["model_cara"] is not None}


VIDEO_EXTENSIONS = (".avi", ".webm", ".mp4", ".mov")


def _count_files(folder, extensions=VIDEO_EXTENSIONS):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(extensions)])


@app.get("/dataset/stats")
def dataset_stats():
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
    stem = os.path.splitext(filename)[0]
    prefix = class_name + "_"
    rest = stem[len(prefix):] if stem.lower().startswith(prefix.lower()) else stem
    if "_" in rest:
        person, _seq = rest.rsplit("_", 1)
        return person
    return None


@app.get("/dataset/recordings")
def dataset_recordings():
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
    safe_filename = os.path.basename(filename)
    safe_word = os.path.basename(word)
    path = os.path.join(SESSIONS_DIR, safe_word, "clips", safe_filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Clip no encontrado.")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=422, detail="No se pudo leer el clip.")

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


_reference_cache = {}


def _pick_reference_clip(palabra):
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

    frames_out = [smoothed[t].reshape(20, 2).tolist() for t in range(smoothed.shape[0])]
    response = {"palabra": safe_palabra, "frames": frames_out}
    _reference_cache[safe_palabra] = {"filename": filename, "response": response}
    return response


DEBUG_FRASES_DIR = os.path.join(BASE_DIR, "data", "debug_frases")
DEBUG_FRASES_MAX_ARCHIVOS = 80


def _guardar_debug_frase(audio_path, recortado_path, palabra, motivo):
    try:
        os.makedirs(DEBUG_FRASES_DIR, exist_ok=True)
        marca = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(DEBUG_FRASES_DIR, f"{marca}_{palabra}_{motivo}")
        if audio_path and os.path.exists(audio_path):
            shutil.copyfile(audio_path, base + "_completo.wav")
        if recortado_path and os.path.exists(recortado_path):
            shutil.copyfile(recortado_path, base + "_recorte.wav")
        archivos = sorted(
            (os.path.join(DEBUG_FRASES_DIR, f) for f in os.listdir(DEBUG_FRASES_DIR)),
            key=os.path.getmtime,
        )
        for viejo in archivos[:-DEBUG_FRASES_MAX_ARCHIVOS]:
            os.remove(viejo)
    except Exception as e:
        print(f"[AVISO] No se pudo guardar el audio de depuracion ({e}).")


def _save_practice_clip(class_name, tmp_path, suffix):
    try:
        clips_dir = os.path.join(SESSIONS_DIR, class_name, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        dest_name = f"{class_name}_web_{int(time.time() * 1000)}{suffix}"
        shutil.copy(tmp_path, os.path.join(clips_dir, dest_name))
        return dest_name
    except OSError as e:
        print(f"[AVISO] No se pudo guardar el clip de práctica ({e}).")
        return None


PREDICT_HARD_TIMEOUT_S = 90


def _score_audio_pronunciation(audio_path, palabra):
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
    clf = _state["audio_clf"]
    if clf is None or audio_path is None or not palabra:
        return None
    class_names = _state["audio_clf_classes"]
    indices_de_la_palabra = [
        i for i, c in enumerate(class_names)
        if c == f"{palabra}_correcto" or c.startswith(f"{palabra}_incorrecto")
    ]
    if not indices_de_la_palabra:
        return None
    try:
        emb = extract_embedding(audio_path, _state["audio_model"], _state["audio_feature_extractor"], DEVICE)
        emb_norm = (emb - _state["audio_clf_mean"]) / _state["audio_clf_std"]
        x = torch.from_numpy(emb_norm.reshape(1, -1)).float().to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(clf(x), dim=1)[0]
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
    audio_dice_bien = audio_clf_resultado["es_correcto"]

    clase_predicha = audio_clf_resultado["clase_predicha"]
    palabra, _pronunciacion_correcta, tipo_error = _parse_clase_predicha(clase_predicha)

    if not audio_confiable:
        veredicto_final = "no_concluyente"
    else:
        veredicto_final = "bien" if audio_dice_bien else "a_practicar"

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

    modelo_video = "cara_completa"
    if palabra_esperada is not None and not (
            f"{palabra_esperada}_correcto" in class_names and f"{palabra_esperada}_incorrecto" in class_names):
        # La palabra no está en el modelo de cara completa (hoy: perro): se evalúa solo con el audio.
        print(f"[predict] '{palabra_esperada}' no está en el modelo de cara completa -> solo audio")
        if audio_path is not None:
            return _run_audio_only_pipeline(audio_path, palabra_esperada)
        return {"valid": False, "reason": f"no hay modelo de video para '{palabra_esperada}' -- mandá también el audio"}

    landmarks_por_frame = detect_landmarks_batch(landmarker, frames)
    print(f"[predict] landmarks detectados ({time.time() - t_start:.1f}s)")

    positions = []
    detected = 0
    last_valid_vector = None
    for landmarks_px in landmarks_por_frame:
        vector = normalize_face_landmarks(landmarks_px) if landmarks_px is not None else None
        if vector is not None and is_outlier(vector, last_valid_vector):
            vector = None
        if vector is not None:
            detected += 1
            last_valid_vector = vector
        positions.append(vector)

    detection_rate = detected / total if total else 0
    filled = fill_missing_frames(positions)
    if filled is None or detection_rate < MIN_DETECCION_CARA:
        return {
            "valid": False,
            "reason": f"cara entera detectada solo en {detected}/{total} frames "
                      f"({detection_rate*100:.0f}%) -- encuadre malo, repetí la toma",
        }

    # [LABIOS DESACTIVADO] positions = []
    # [LABIOS DESACTIVADO] detected = 0
    # [LABIOS DESACTIVADO] last_valid_vector = None
    # [LABIOS DESACTIVADO] for landmarks_px in landmarks_por_frame:
        # [LABIOS DESACTIVADO] vector = normalize_lip_landmarks(landmarks_px) if landmarks_px is not None else None
        # [LABIOS DESACTIVADO] if vector is not None and is_outlier(vector, last_valid_vector):
            # [LABIOS DESACTIVADO] vector = None
        # [LABIOS DESACTIVADO] if vector is not None:
            # [LABIOS DESACTIVADO] detected += 1
            # [LABIOS DESACTIVADO] last_valid_vector = vector
        # [LABIOS DESACTIVADO] positions.append(vector)

    # [LABIOS DESACTIVADO] detection_rate = detected / total if total else 0
    # [LABIOS DESACTIVADO] filled = fill_missing_frames(positions)
    # [LABIOS DESACTIVADO] if filled is None or detection_rate < MIN_DETECTION_RATE_VALID:
        # [LABIOS DESACTIVADO] return {
            # [LABIOS DESACTIVADO] "valid": False,
            # [LABIOS DESACTIVADO] "reason": f"cara detectada solo en {detected}/{total} frames "
                      # [LABIOS DESACTIVADO] f"({detection_rate*100:.0f}%) -- encuadre malo, repetí la toma",
        # [LABIOS DESACTIVADO] }

    # [LABIOS DESACTIVADO] modelo_video = "labios"
    # [LABIOS DESACTIVADO] clases_cara = _state["class_names_cara"]
    # [LABIOS DESACTIVADO] if (_state["model_cara"] is not None and palabra_esperada is not None
            # [LABIOS DESACTIVADO] and f"{palabra_esperada}_correcto" in clases_cara and f"{palabra_esperada}_incorrecto" in clases_cara):
        # [LABIOS DESACTIVADO] posiciones_cara = []
        # [LABIOS DESACTIVADO] detectados_cara = 0
        # [LABIOS DESACTIVADO] ultimo_cara = None
        # [LABIOS DESACTIVADO] for landmarks_px in landmarks_por_frame:
            # [LABIOS DESACTIVADO] vector_cara = normalize_face_landmarks(landmarks_px) if landmarks_px is not None else None
            # [LABIOS DESACTIVADO] if vector_cara is not None and is_outlier(vector_cara, ultimo_cara):
                # [LABIOS DESACTIVADO] vector_cara = None
            # [LABIOS DESACTIVADO] if vector_cara is not None:
                # [LABIOS DESACTIVADO] detectados_cara += 1
                # [LABIOS DESACTIVADO] ultimo_cara = vector_cara
            # [LABIOS DESACTIVADO] posiciones_cara.append(vector_cara)
        # [LABIOS DESACTIVADO] tasa_cara = detectados_cara / total
        # [LABIOS DESACTIVADO] relleno_cara = fill_missing_frames(posiciones_cara)
        # [LABIOS DESACTIVADO] if relleno_cara is not None and tasa_cara >= MIN_DETECCION_CARA:
            # [LABIOS DESACTIVADO] filled = relleno_cara
            # [LABIOS DESACTIVADO] model = _state["model_cara"]
            # [LABIOS DESACTIVADO] class_names = clases_cara
            # [LABIOS DESACTIVADO] max_frames = _state["max_frames_cara"]
            # [LABIOS DESACTIVADO] modelo_video = "cara_completa"
        # [LABIOS DESACTIVADO] else:
            # [LABIOS DESACTIVADO] print(f"[predict] cara completa no usable ({detectados_cara}/{total} frames con la cara entera): uso el modelo de labios")
    sequence, mask = preprocess_sequence(filled, max_frames)
    sequence, mask = sequence.to(DEVICE), mask.to(DEVICE)
    with torch.inference_mode():
        logits = model(sequence, src_key_padding_mask=mask)
        probs = torch.softmax(logits, dim=1)[0]
    print(f"[predict] modelo Conformer ({modelo_video}) listo ({time.time() - t_start:.1f}s total)")

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    top_idx = int(sorted_idx[0].item())
    top_prob = float(sorted_probs[0].item())
    second_prob = float(sorted_probs[1].item()) if len(sorted_probs) > 1 else 0.0
    margin = top_prob - second_prob
    clase_predicha = class_names[top_idx]
    print(f"[predict] video[{modelo_video}]: clase={clase_predicha} prob={top_prob:.2f} margin={margin:.2f} "
          f"(2da opcion: {class_names[int(sorted_idx[1].item())] if len(sorted_idx) > 1 else '?'} "
          f"{second_prob:.2f}) | esperado={palabra_esperada}")
    palabra, pronunciacion_correcta, tipo_error = _parse_clase_predicha(clase_predicha)

    def _palabra_base(nombre):
        return nombre.split("_")[0]

    palabra_distinta = False
    if palabra_esperada is not None and palabra != palabra_esperada:
        pidio_palabra_completa = palabra_esperada == _palabra_base(palabra_esperada)
        misma_palabra_base = _palabra_base(palabra) == _palabra_base(palabra_esperada)
        if not (pidio_palabra_completa and misma_palabra_base):
            pronunciacion_correcta = False
            tipo_error = None
            palabra_distinta = True

    all_probs = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

    es_palabra_completa = (palabra_esperada or palabra) and "_" not in (palabra_esperada or palabra)
    audio_resultado = (
        _score_audio_pronunciation(audio_path, palabra_esperada or palabra)
        if es_palabra_completa else None
    )
    audio_ok = (
        audio_resultado is not None
        and len(audio_resultado["sospechosos"]) == 0
        and audio_resultado["palabra_reconocida"]
    )

    audio_clf_resultado = _classify_audio(audio_path, palabra_esperada or palabra)
    prob_str = f"{audio_clf_resultado['prob']:.2f}" if audio_clf_resultado else "?"
    print(f"[predict] audio GOP {'OK' if audio_resultado is None else ('bien' if audio_ok else 'sospechoso')} | "
          f"clasificador {'sin entrenar' if audio_clf_resultado is None else audio_clf_resultado['clase_predicha']} "
          f"prob={prob_str} (umbral {AUDIO_MIN_CONFIDENCE}) video_prob={top_prob:.2f} "
          f"({time.time() - t_start:.1f}s total)")

    acoustic_match = (
        audio_clf_resultado is not None
        and audio_clf_resultado["prob"] >= ACOUSTIC_MATCH_THRESHOLD
        and audio_clf_resultado["es_correcto"]
    )
    visual_match = top_prob >= VISUAL_MATCH_THRESHOLD and pronunciacion_correcta

    confianza_suficiente = audio_clf_resultado is not None

    audio_gop_ok = (
        audio_resultado is not None
        and not audio_resultado["sospechosos"]
        and audio_resultado["palabra_reconocida"]
    )
    audio_confirma = (
        audio_clf_resultado is not None
        and audio_clf_resultado["es_correcto"]
        and (audio_clf_resultado["prob"] >= AUDIO_MIN_CONFIDENCE
             or (audio_gop_ok and audio_clf_resultado["prob"] >= AUDIO_CONFIRMA_CON_GOP))
    )
    video_dice_mal_misma_palabra = (
        (not palabra_distinta) and top_prob >= VISUAL_MATCH_THRESHOLD and not pronunciacion_correcta
    )

    if not confianza_suficiente:
        veredicto_final = "no_concluyente"
        tipo_error = None
    elif acoustic_match and visual_match:
        veredicto_final = "bien"
        tipo_error = None
    elif audio_confirma and not video_dice_mal_misma_palabra:
        veredicto_final = "bien"
        tipo_error = None
        print(f"[predict] audio confirma '{palabra_esperada}' (clasificador {audio_clf_resultado['prob']:.2f}, "
              f"GOP {'bien' if audio_gop_ok else 'dudoso'}); el video ({palabra}, {top_prob:.2f}) no lo contradice "
              f"con seguridad -> bien")
    elif (not acoustic_match) and visual_match:
        veredicto_final = "a_practicar"
        tipo_error = "sustitucion_acustica"
    elif (not acoustic_match) and (not visual_match):
        veredicto_final = "a_practicar"
        tipo_error = "motor_severo"
    else:
        veredicto_final = "a_practicar"
        tipo_error = "compensacion_visual"

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
        "tipo_error": tipo_error if (confianza_suficiente and veredicto_final == "a_practicar") else None,
        "palabra_distinta": palabra_distinta and veredicto_final == "a_practicar",
        "class_name": clase_predicha,
        "prob": top_prob,
        "detected": detected,
        "total": total,
        "modelo_video": modelo_video,
        "confianza_suficiente": confianza_suficiente,
        "capture_file": capture_file,
        "all_probs": all_probs,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(None),
    audio: UploadFile = File(None),
    palabra: str = Form(None),
):
    if _state["model_cara"] is None:
        raise HTTPException(status_code=503, detail="El servidor todavía está cargando el modelo.")

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
    model = _state["model_cara"]
    class_names = _state["class_names_cara"]
    max_frames = _state["max_frames_cara"]

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
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


FRASE_DISTANCIA_MAXIMA = 0.48


def _distancia_frase(frase, resultado):
    try:
        import audio_pronunciation as _ap
        _ids, usados = _ap._phoneme_to_token_ids(_ap.texto_a_fonemas(frase), _state["audio_vocab"])
        reconocidos = resultado["reconocido_libre"].split()
        if not usados:
            return None
        d = _ap._distancia_edicion(usados, reconocidos)
        return d / max(len(usados), len(reconocidos), 1)
    except Exception:
        return None


@app.post("/predict_frase")
async def predict_frase(
    audio: UploadFile = File(...),
    frase: str = Form(...),
    palabra: str = Form(...),
):
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

        if not resultado.get("frase_completa_dicha", True):
            _guardar_debug_frase(audio_path, audio_recortado_path, palabra, "incompleta")
            return {
                "valid": False,
                "frase_incompleta": True,
                "reason": "decí toda la frase, despacito -- ¡vos podés!",
            }

        dist_frase = _distancia_frase(frase, resultado)
        if dist_frase is None:
            frase_ok = resultado.get("resto_de_la_frase_ok", True)
        else:
            frase_ok = dist_frase <= FRASE_DISTANCIA_MAXIMA
        if not frase_ok:
            _guardar_debug_frase(audio_path, audio_recortado_path, palabra,
                                 f"distinta_d{'' if dist_frase is None else round(dist_frase, 2)}")
            return {
                "valid": False,
                "frase_distinta": True,
                "reason": "repetí la frase de arriba tal cual -- ¡dale, de nuevo!",
            }

        clf_resultado = _classify_audio(audio_recortado_path, palabra) if audio_recortado_path else None

        gop_ok = len(resultado["sospechosos"]) == 0 and resultado["palabra_reconocida"]

        clf_confiable = clf_resultado is not None and clf_resultado["prob"] >= AUDIO_MIN_CONFIDENCE
        if clf_confiable:
            veredicto_final = "bien" if clf_resultado["es_correcto"] else "a_practicar"
            fuente = "clasificador"
        else:
            veredicto_final = "bien" if gop_ok else "a_practicar"
            fuente = "gop"
        tipo_error = resultado.get("tipo_error_audio") if veredicto_final == "a_practicar" else None

        if veredicto_final == "a_practicar" or not clf_confiable:
            _prob_dbg = f"{clf_resultado['prob']:.2f}" if clf_resultado else "sinclf"
            _guardar_debug_frase(audio_path, audio_recortado_path, palabra,
                                 f"{veredicto_final}_{fuente}_clf{_prob_dbg}")

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


def _ciclos_pataka(tokens):
    siguiente = {"p": "t", "t": "k", "k": "p"}
    esperado, ciclos = "p", 0
    for tok in tokens:
        base = tok[:1]
        if base == esperado:
            if esperado == "k":
                ciclos += 1
            esperado = siguiente[esperado]
    return ciclos


_PALABRAS_DE_RELLENO = {"en", "un", "una", "el", "la", "los", "las", "de", "del", "y", "e", "o", "a", "al", "con", "por"}
_UMBRAL_SONIDO_MAL = -4.0


def _palabras_flojas(texto, resultado):
    try:
        import audio_pronunciation as _ap
        vocab = _state["audio_vocab"]
        palabras, pos, malas = texto.split(), 0, []
        for pal in palabras:
            _ids, usados = _ap._phoneme_to_token_ids(_ap.texto_a_fonemas(pal), vocab)
            n = len(usados)
            if n == 0:
                continue
            if pal.lower() not in _PALABRAS_DE_RELLENO:
                scores = resultado["scores"][pos:pos + n]
                if any(sc is None or sc < _UMBRAL_SONIDO_MAL for sc in scores):
                    malas.append(pal)
            pos += n
        if pos != len(resultado["fonemas"]):
            return None
        return malas
    except Exception:
        return None


@app.post("/ejercicio_resultado")
def ejercicio_resultado(
    ejercicio_id: str = Form(...),
    tipo: str = Form(...),
    correcto: bool = Form(...),
    fuerza: int = Form(None),
    aguante: int = Form(None),
    parejo: int = Form(None),
    voz: int = Form(None),
    porcentaje_ok: float = Form(None),
    tipo_error: str = Form(None),
):
    if tipo not in ("praxia", "soplo", "voz", "practica", "diagnostico", "evaluacion", "palabra"):
        raise HTTPException(status_code=400, detail="tipo tiene que ser praxia, soplo, voz, practica, diagnostico, evaluacion o palabra.")
    for nombre, valor in (("fuerza", fuerza), ("aguante", aguante), ("parejo", parejo), ("voz", voz)):
        if valor is not None and not 1 <= valor <= 3:
            raise HTTPException(status_code=400, detail=f"{nombre} va de 1 a 3 estrellas.")
    guardado = db.save_ejercicio(
        ejercicio_id=ejercicio_id[:40], tipo=tipo, correcto=correcto, fuerza=fuerza, aguante=aguante,
        parejo=parejo, voz=voz, porcentaje_ok=porcentaje_ok, tipo_error=tipo_error,
    )
    return {"guardado": guardado}


@app.get("/ejercicios/resumen")
def ejercicios_resumen():
    filas = db.get_ejercicios_resumen()
    return {"ejercicios": filas if filas is not None else [], "base_de_datos": filas is not None}


@app.post("/ejercicio_voz")
async def ejercicio_voz(
    audio: UploadFile = File(...),
    texto: str = Form(...),
    tipo: str = Form("trabalenguas"),
):
    if _state["audio_model"] is None:
        raise HTTPException(status_code=503, detail="El servidor todavía está cargando el modelo de audio.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        shutil.copyfileobj(audio.file, tmp_audio)
        audio_path = tmp_audio.name

    try:
        waveform = load_audio_16k(audio_path)
        pico = float(waveform.abs().max()) if waveform.numel() else 0.0
        if pico < 0.01:
            return {"valid": False, "reason": "no te escuché -- ¡probá de nuevo, más fuerte!"}
        activo = (waveform.abs() > 0.05 * pico).nonzero()
        seg_habla = max(float(activo[-1] - activo[0]) / TARGET_SAMPLE_RATE, 0.5) if len(activo) else 0.5

        resultado = _score_audio_pronunciation(audio_path, texto)
        if resultado is None:
            return {"valid": False, "reason": "no se pudo analizar el audio -- probá de nuevo"}

        scores = resultado["scores"]
        flojos = set(resultado["sospechosos"])
        ok = [i for i, sc in enumerate(scores) if sc is not None and i not in flojos]
        pct_ok = len(ok) / max(len(scores), 1)

        salida = {
            "valid": True,
            "tipo": tipo,
            "porcentaje_ok": round(100 * pct_ok),
            "palabra_reconocida": resultado["palabra_reconocida"],
            "segundos_de_habla": round(seg_habla, 1),
            "tipo_error": resultado.get("tipo_error_audio"),
            "fonemas": resultado["fonemas"],
            "scores": [None if s is None else round(float(s), 2) for s in resultado["scores"]],
            "reconocido_libre": resultado.get("reconocido_libre"),
        }
        tokens = resultado["reconocido_libre"].split()
        if tipo == "ritmo":
            ciclos = _ciclos_pataka(tokens)
            silabas = sum(1 for t in tokens if t[:1] in ("p", "t", "k"))
            salida.update({
                "ciclos": ciclos,
                "silabas": silabas,
                "silabas_por_segundo": round(silabas / seg_habla, 1),
                "correcta": ciclos >= 3,
            })
        else:
            malas = _palabras_flojas(texto, resultado)
            salida["palabras_flojas"] = malas if malas is not None else []
            # Palabras sueltas cortas (3-4 sonidos): un solo sonido dudoso baja a 67 %, así que
            # el corte es más flexible que en frases y trabalenguas.
            corte = 0.7 if tipo == "palabra" else 0.75
            if tipo == "palabra":
                # Palabra suelta: además del porcentaje, un sonido muy mal dicho tiene que fallar la palabra.
                #  - el PRIMER sonido (el que casi siempre se cambia: "tan" por "pan") no puede bajar de -3.5,
                #    salvo en las palabras que empiezan con n, ñ, y, ll o ch, que este modelo puntúa raro aunque
                #    estén bien dichas (en ellas el primer sonido no se evalúa);
                #  - ningún otro sonido puede bajar de -7 ("dato" por "gato", "supete" por "chupete").
                # Se endureció (antes -4.5 y -9) porque palabras mal dichas con voz real se aceptaban como bien;
                # falta calibrar con niños. Cada intento se anota en data/intentos_voz.csv para ajustar los umbrales.
                inicio_inestable = texto.strip().lower().startswith(("ñ", "y", "ll", "ch"))
                # La N inicial sí se mide bien (bien dicha ~0; si la omiten, "ido" por "nido", baja a -6 o menos).
                inicio_n = texto.strip().lower().startswith("n")
                evaluados = scores[1:] if inicio_inestable else scores
                validos = [s for s in evaluados if s is not None]
                primero = scores[0] if scores else None
                primer_sonido_mal = (primero is not None and primero < -3.5 and not inicio_inestable)
                algun_sonido_muy_mal = (bool(validos) and min(validos) < -7.0) or (
                    inicio_inestable and primero is not None and primero < -9.0) or (
                    inicio_n and primero is not None and primero < -5.0)
                salida["correcta"] = pct_ok >= corte and not primer_sonido_mal and not algun_sonido_muy_mal
            else:
                salida["correcta"] = (
                    pct_ok >= corte
                    and bool(resultado["palabra_reconocida"])
                    and not malas
                )
        palabra_corta = "".join(ch for ch in texto.lower() if ch.isalnum())[:14]
        _guardar_debug_frase(audio_path, None, f"ej_{tipo}_{palabra_corta}", f"ok{salida['porcentaje_ok']}_{'bien' if salida['correcta'] else 'mal'}")
        try:
            ruta_log = os.path.join(BASE_DIR, "data", "intentos_voz.csv")
            nuevo_log = not os.path.exists(ruta_log)
            with open(ruta_log, "a", encoding="utf-8") as fh:
                if nuevo_log:
                    fh.write("fecha;palabra;tipo;porcentaje_ok;scores;veredicto\n")
                fh.write(f"{datetime.now().isoformat(timespec='seconds')};{texto};{tipo};{salida['porcentaje_ok']};"
                         f"{salida.get('scores')};{'bien' if salida['correcta'] else 'mal'}\n")
        except Exception:
            pass

        print(f"[ejercicio-voz] tipo={tipo} texto={texto!r} | ok={salida['porcentaje_ok']}% "
              f"reconocida={salida['palabra_reconocida']} | correcta={salida['correcta']}"
              + (f" ciclos={salida['ciclos']}" if tipo == "ritmo" else "")
              + (f" flojas={salida.get('palabras_flojas')}" if tipo != "ritmo" else ""))
        return salida
    except Exception as e:
        return {"valid": False, "reason": f"no se pudo analizar el audio ({e}) -- probá de nuevo"}
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn

    pid_file = os.path.join(BASE_DIR, "server", ".server.pid")
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    uvicorn.run(app, host="127.0.0.1", port=8000)
