"""
build_ood_centroids.py
-----------------------
El modelo (LipReadingConformer) es un clasificador de conjunto CERRADO: se
entrenó SOLO con las frases de bullying, nunca vio ejemplos de habla normal
("hola qué tal", silencio, etc.) -- así que frente a algo que no reconoce, el
softmax puede salir igual de confiado que con una frase real (confirmado en
producción: "hola qué tal" se clasificó como una frase de riesgo con más de
90% de confianza). Subir el umbral de probabilidad NO alcanza para arreglar
esto de raíz.

Este script agrega una segunda señal, independiente del softmax: para cada
clase entrenada, calcula el CENTROIDE (promedio) del vector de features
"pooled" (justo antes del clasificador, ver LipReadingConformer.forward_
features en train_landmarks_transformer.py) de todos los clips de esa clase
en el dataset de entrenamiento, más un radio de aceptación (percentil 99 de
la distancia de cada clip de esa clase a su propio centroide).

En la predicción real (server/main.py), además de mirar el softmax, se mide
la distancia del clip nuevo al centroide de la clase que el modelo eligió.
Si esa distancia es mucho mayor que lo que se vio en TODO el dataset de
entrenamiento para esa clase, es una señal fuerte de que el clip no se
parece a ningún ejemplo real de esa frase -- aunque el softmax diga que sí.

Uso:
    python scripts/build_ood_centroids.py

Requiere haber corrido antes: python scripts/extraer_landmarks_npy.py
Escribe en: models/ood_centroids.npz
"""
import glob
import os

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "landmarks_npy")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mejor_modelo_landmarks_conformer.pth")
OUT_PATH = os.path.join(BASE_DIR, "models", "ood_centroids.npz")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def build():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_landmarks_transformer import LipReadingConformer, make_padding_mask

    checkpoint = safe_torch_load(MODEL_PATH)
    class_names = checkpoint["class_names"]
    max_frames = checkpoint["max_frames"]
    input_dim = checkpoint.get("input_dim", 240)

    model = LipReadingConformer(num_classes=len(class_names), input_dim=input_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    centroids = np.zeros((len(class_names), model.classifier[0].in_features), dtype=np.float32)
    radii = np.zeros(len(class_names), dtype=np.float32)

    print(f"Calculando centroides de {len(class_names)} clases (device={DEVICE})...")
    with torch.inference_mode():
        for ci, word in enumerate(class_names):
            files = glob.glob(os.path.join(DATASET_DIR, word, "*.npy"))
            if not files:
                print(f"  [AVISO] {word}: sin clips en {DATASET_DIR}, centroide queda en cero.")
                continue

            pooled_list = []
            batch_size = 64
            for start in range(0, len(files), batch_size):
                batch_files = files[start:start + batch_size]
                seqs, lengths = [], []
                for f in batch_files:
                    seq = np.load(f).astype(np.float32)  # (T, input_dim)
                    T = seq.shape[0]
                    if T < max_frames:
                        pad = np.zeros((max_frames - T, seq.shape[1]), dtype=np.float32)
                        seq = np.concatenate([seq, pad], axis=0)
                        lengths.append(T)
                    else:
                        seq = seq[:max_frames]
                        lengths.append(max_frames)
                    seqs.append(seq)
                batch = torch.from_numpy(np.stack(seqs)).to(DEVICE)          # (B, max_frames, input_dim)
                lengths_t = torch.tensor(lengths, device=DEVICE)
                mask = make_padding_mask(lengths_t, max_frames, DEVICE)
                pooled = model.forward_features(batch, src_key_padding_mask=mask)  # (B, hidden_dim)
                pooled_list.append(pooled.cpu().numpy())

            pooled_all = np.concatenate(pooled_list, axis=0)  # (N, hidden_dim)
            centroid = pooled_all.mean(axis=0)
            distances = np.linalg.norm(pooled_all - centroid, axis=1)
            radius = float(np.percentile(distances, 99))

            centroids[ci] = centroid
            radii[ci] = radius
            print(f"  {word:28s} {len(files):4d} clips -- radio (p99) = {radius:.3f}")

    np.savez(OUT_PATH, class_names=np.array(class_names), centroids=centroids, radii=radii)
    print(f"\nGuardado: {OUT_PATH}")


if __name__ == "__main__":
    build()
