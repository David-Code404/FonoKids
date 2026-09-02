"""
lip_reading_dataset.py
--------------------------
PARTE 2 del pipeline Transformer-puro: Dataset y DataLoader de PyTorch que
leen los .npy generados por extraer_landmarks_npy.py (secuencias de 80
landmarks normalizados por frame).

Cada video tiene una cantidad de frames distinta -- acá se hace padding con
ceros (o truncado) para que todas las secuencias queden en max_frames fijos,
y se devuelve también la longitud real (útil si más adelante querés usar
attention masks o packed sequences).
"""
import os
import glob
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "landmarks_npy"
)
LANDMARK_DIM = 80  # 40 puntos de labios x (x, y)


class LipReadingDataset(Dataset):
    """Lee data/landmarks_npy/<palabra>/*.npy y arma (secuencia, label).

    Args:
        data_dir: carpeta con subcarpetas por palabra (label_map.json adentro).
        max_frames: longitud fija de secuencia (padding con ceros o truncado).
    """

    def __init__(self, data_dir=DEFAULT_DATA_DIR, max_frames=60):
        self.data_dir = data_dir
        self.max_frames = max_frames

        label_map_path = os.path.join(data_dir, "label_map.json")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(
                f"No encontré {label_map_path}. Corré extraer_landmarks_npy.py primero."
            )
        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)

        num_classes = max(self.label_map.values()) + 1
        self.class_names = ["?"] * num_classes
        for word, idx in self.label_map.items():
            self.class_names[idx] = word

        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*", "*.npy")))
        if not self.file_paths:
            raise RuntimeError(f"No hay archivos .npy en {data_dir}.")

    def __len__(self):
        return len(self.file_paths)

    def _label_from_path(self, path):
        word = os.path.basename(os.path.dirname(path))
        return self.label_map[word]

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        sequence = np.load(path).astype(np.float32)  # (T, 80)
        label = self._label_from_path(path)

        T = sequence.shape[0]
        if T < self.max_frames:
            pad = np.zeros((self.max_frames - T, LANDMARK_DIM), dtype=np.float32)
            sequence = np.concatenate([sequence, pad], axis=0)
            real_length = T
        elif T > self.max_frames:
            sequence = sequence[: self.max_frames]
            real_length = self.max_frames
        else:
            real_length = T

        return (
            torch.from_numpy(sequence),                  # (max_frames, 80)
            torch.tensor(label, dtype=torch.long),
            torch.tensor(real_length, dtype=torch.long),  # frames reales antes del padding
        )


def build_dataloaders(data_dir=DEFAULT_DATA_DIR, max_frames=60, batch_size=32,
                       val_ratio=0.2, seed=42, num_workers=0):
    """Arma el dataset completo y lo separa en train/val, devolviendo los
    DataLoaders listos junto con el dataset (para poder leer class_names)."""
    dataset = LipReadingDataset(data_dir=data_dir, max_frames=max_frames)

    n = len(dataset)
    val_size = max(1, int(n * val_ratio))
    train_size = n - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)

    return dataset, train_loader, val_loader


if __name__ == "__main__":
    # Chequeo rápido de que todo lee bien -- no entrena nada.
    dataset, train_loader, val_loader = build_dataloaders()
    print(f"Clases ({len(dataset.class_names)}): {dataset.class_names}")
    print(f"Total muestras: {len(dataset)} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    sequences, labels, lengths = next(iter(train_loader))
    print(f"Batch de ejemplo -> sequences: {sequences.shape}, labels: {labels.shape}, lengths: {lengths.shape}")
