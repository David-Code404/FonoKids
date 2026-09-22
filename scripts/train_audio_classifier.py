"""
train_audio_classifier.py
--------------------------
Entrena un clasificador de AUDIO real, con TUS clips grabados (los .wav que
graba grabar_video_continuo.py junto a cada .avi) -- a diferencia del GOP de
audio_pronunciation.py (que usa el modelo wav2vec2 genérico "tal cual viene",
sin entrenamiento propio), esto aprende directamente de tus datos, igual
criterio que train_landmarks_transformer.py con los landmarks visuales.

Por qué hace falta esto además del GOP: el GOP fuerza el audio contra los
fonemas de la palabra esperada pase lo que pase, así que a veces no detecta
que se dijo OTRA palabra (confirmado: "pedo" en vez de "perro" no siempre
disparaba la alarma). Un clasificador entrenado con ejemplos reales de cada
clase (correcto/incorrecto de cada tipo) aprende a distinguir eso de forma
directa, en vez de depender de una alineación forzada.

Pipeline:
    1. Extrae un embedding (vector fijo de 1024 números, ver
       audio_pronunciation.extract_embedding) por cada clip .wav de
       data/sesiones_continuas/<clase>/clips/ -- se cachea en
       data/audio_embeddings_cache/ para no recalcular en cada corrida.
    2. Separa en train/val (80/20, estratificado por clase).
    3. Entrena una red chica (Linear -> ReLU -> Dropout -> Linear) sobre los
       embeddings -- no hace falta más, ya es una representación rica.
    4. Reporta accuracy real de validación (sin inventar números).
    5. Guarda models/audio_classifier.pth.

Uso:
    python train_audio_classifier.py
"""
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_pronunciation import EMBEDDING_DIM, extract_embedding, load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")
CACHE_DIR = os.path.join(BASE_DIR, "data", "audio_embeddings_cache")
MODEL_OUT = os.path.join(BASE_DIR, "models", "audio_classifier.pth")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

VAL_FRACTION = 0.2
SEED = 42
EPOCHS = 150
HIDDEN_DIM = 128
DROPOUT = 0.4
WEIGHT_DECAY = 1e-3
LR = 1e-3
EARLY_STOPPING_PATIENCE = 15


def find_labeled_clips():
    """{clase: [ruta.wav, ...]} para cada carpeta <palabra>_correcto o
    <palabra>_incorrecto[_subtipo] que tenga clips con audio.

    Fallback: si no existe data/sesiones_continuas (ej. se bajaron a esta PC
    solo los embeddings ya calculados en Colab, sin los .wav originales),
    reconstruye las clases a partir de los nombres de archivo ya cacheados
    en data/audio_embeddings_cache/ (formato "<clase>_NNNN.npy") -- en ese
    caso se entrena directo desde el cache, sin necesitar los .wav."""
    clips_por_clase = {}
    if os.path.isdir(SESSIONS_DIR):
        for clase in sorted(os.listdir(SESSIONS_DIR)):
            clips_dir = os.path.join(SESSIONS_DIR, clase, "clips")
            if not os.path.isdir(clips_dir):
                continue
            wavs = sorted(
                os.path.join(clips_dir, f) for f in os.listdir(clips_dir) if f.lower().endswith(".wav")
            )
            if wavs:
                clips_por_clase[clase] = wavs
        if clips_por_clase:
            return clips_por_clase

    if os.path.isdir(CACHE_DIR):
        print(f"[AVISO] No encontré {SESSIONS_DIR} -- entrenando directo desde "
              f"los embeddings ya cacheados en {CACHE_DIR}.")
        por_clase = {}
        for f in sorted(os.listdir(CACHE_DIR)):
            if not f.endswith(".npy"):
                continue
            clase = re.sub(r"_\d+$", "", os.path.splitext(f)[0])
            por_clase.setdefault(clase, []).append(os.path.join(CACHE_DIR, f))
        return por_clase

    return clips_por_clase


def build_embeddings(clips_por_clase, model, feature_extractor, device):
    """Extrae (o lee de la caché) el embedding de cada clip. Devuelve
    (X, y, class_names) -- X: (N, 1024) float32, y: (N,) int, class_names:
    lista ordenada usada para mapear y -> nombre de clase.

    Los items de clips_por_clase pueden ser rutas a .wav (caso normal) o
    directo a un .npy ya cacheado (caso fallback sin sesiones_continuas,
    ver find_labeled_clips) -- en ese caso se carga tal cual, sin necesitar
    el modelo de wav2vec2."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    class_names = sorted(clips_por_clase.keys())
    X, y = [], []
    total = sum(len(v) for v in clips_por_clase.values())
    done = 0
    for clase in class_names:
        for item_path in clips_por_clase[clase]:
            if item_path.lower().endswith(".npy"):
                emb = np.load(item_path)
            else:
                cache_name = os.path.splitext(os.path.basename(item_path))[0] + ".npy"
                cache_path = os.path.join(CACHE_DIR, cache_name)
                if os.path.exists(cache_path):
                    emb = np.load(cache_path)
                else:
                    emb = extract_embedding(item_path, model, feature_extractor, device)
                    np.save(cache_path, emb)
            X.append(emb)
            y.append(class_names.index(clase))
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  embeddings: {done}/{total}")
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64), class_names


def stratified_split(y, val_fraction, seed):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx), np.array(val_idx)


class AudioClassifierHead(nn.Module):
    """Cabeza chica sobre el embedding congelado de wav2vec2 -- con ~200
    clips en total y un embedding de 1024, una red más grande solo
    sobreajustaría. Dropout + weight_decay como regularización principal."""

    def __init__(self, in_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class EarlyStopping:
    """Corta el entrenamiento si val_loss deja de mejorar por `patience`
    epocas seguidas -- mismo criterio que usa train_landmarks_transformer.py
    para el modelo visual."""

    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r--", label="Validation Loss", linewidth=2)
    ax1.set_title("Perdida (audio)")
    ax1.set_xlabel("Epocas"); ax1.set_ylabel("Perdida"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "g-", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Precision del clasificador de audio")
    ax2.set_xlabel("Epocas"); ax2.set_ylabel("Precision (%)"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "audio_classifier_curvas.png")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"-> Grafico guardado: {out_path}")


def plot_confusion_matrix(conf, class_names, val_acc):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.45), max(5, n * 0.45)))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion (audio) -- val acc {val_acc*100:.1f}%")
    plt.colorbar(im)
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "audio_matriz_confusion.png")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"-> Matriz de confusion guardada: {out_path}")


def main():
    clips_por_clase = find_labeled_clips()
    if not clips_por_clase:
        print(f"No encontré clips con audio en {SESSIONS_DIR}. Grabá primero con "
              "grabar_video_continuo.py.")
        return
    print("Clases encontradas:")
    for clase, wavs in clips_por_clase.items():
        print(f"  {clase}: {len(wavs)} clips")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    todos_cacheados = all(
        item.lower().endswith(".npy") for wavs in clips_por_clase.values() for item in wavs
    )
    if todos_cacheados:
        print("\nTodos los embeddings ya están cacheados -- no hace falta cargar wav2vec2.")
        model = feature_extractor = None
    else:
        print(f"\nCargando wav2vec2 para extraer embeddings (device={device})...")
        model, feature_extractor, _vocab, device = load_model(device=device)

    print("\nExtrayendo embeddings (con caché, las corridas siguientes son instantáneas)...")
    X, y, class_names = build_embeddings(clips_por_clase, model, feature_extractor, device)
    print(f"\nTotal: {len(X)} clips, {len(class_names)} clases: {class_names}")

    train_idx, val_idx = stratified_split(y, VAL_FRACTION, SEED)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    # Normalización simple (media/desvío del set de train) -- ayuda a que la
    # red chica converja más estable, nada exótico.
    mean = X[train_idx].mean(axis=0, keepdims=True)
    std = X[train_idx].std(axis=0, keepdims=True) + 1e-6

    def to_tensor(idx):
        xs = (X[idx] - mean) / std
        return torch.from_numpy(xs).float().to(device), torch.from_numpy(y[idx]).long().to(device)

    x_train, y_train = to_tensor(train_idx)
    x_val, y_val = to_tensor(val_idx)

    head = AudioClassifierHead(EMBEDDING_DIM, HIDDEN_DIM, len(class_names), DROPOUT).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=0.001)
    for epoch in range(1, EPOCHS + 1):
        head.train()
        optimizer.zero_grad()
        logits = head(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            train_acc = (head(x_train).argmax(1) == y_train).float().mean().item()
            val_logits = head(x_val)
            val_loss = criterion(val_logits, y_val).item()
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc * 100)
        history["val_acc"].append(val_acc * 100)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | loss {loss.item():.3f} | val_loss {val_loss:.3f} | "
                  f"train_acc {train_acc:.2f} | val_acc {val_acc:.2f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"  -> Early stopping activado en la epoca {epoch} "
                  f"(sin mejora en val_loss por {EARLY_STOPPING_PATIENCE} epocas).")
            break

    plot_training_curves(history)

    print(f"\nMejor accuracy de validación: {best_val_acc:.2f} ({int(best_val_acc*len(val_idx))}/{len(val_idx)})")
    if len(val_idx) < 10:
        print("[AVISO] Set de validación muy chico (<10 clips) -- este número es orientativo, "
              "no una medición confiable todavía. Va a mejorar con más datos.")

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_pred = head(x_val).argmax(1).cpu().numpy()
    val_true = y[val_idx]
    print("\nMatriz de confusión (filas=real, columnas=predicho):")
    print("clases:", class_names)
    n = len(class_names)
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(val_true, val_pred):
        conf[t, p] += 1
    for i, row in enumerate(conf):
        print(f"  {class_names[i]:35s} {row.tolist()}")

    plot_confusion_matrix(conf, class_names, best_val_acc)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "class_names": class_names,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "mean": mean,
            "std": std,
            "val_acc": best_val_acc,
        },
        MODEL_OUT,
    )
    print(f"\nGuardado en {MODEL_OUT}")


if __name__ == "__main__":
    main()
