import os
import glob
import json
import unicodedata

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "landmarks_npy"
)
LANDMARK_DIM = 240


def _nfc(text):
    return unicodedata.normalize("NFC", text)


class LipReadingDataset(Dataset):

    def __init__(self, data_dir=DEFAULT_DATA_DIR, max_frames=60, file_paths=None, augment=False):
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.augment = augment

        label_map_path = os.path.join(data_dir, "label_map.json")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(
                f"No encontré {label_map_path}. Corré extraer_landmarks_npy.py primero."
            )
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw_label_map = json.load(f)
        self.label_map = {_nfc(word): idx for word, idx in raw_label_map.items()}

        num_classes = max(self.label_map.values()) + 1
        self.class_names = ["?"] * num_classes
        for word, idx in self.label_map.items():
            self.class_names[idx] = word

        if file_paths is not None:
            self.file_paths = list(file_paths)
        else:
            self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*", "*.npy")))
        if not self.file_paths:
            raise RuntimeError(f"No hay archivos .npy en {data_dir}.")

        first_sample = np.load(self.file_paths[0])
        self.landmark_dim = int(first_sample.shape[-1])

    def __len__(self):
        return len(self.file_paths)

    def _label_from_path(self, path):
        word = _nfc(os.path.basename(os.path.dirname(path)))
        return self.label_map[word]

    def _apply_augmentation(self, sequence):
        sequence = sequence + np.random.normal(0, 0.003, sequence.shape).astype(np.float32)

        shift = np.random.uniform(-0.01, 0.01, size=(1, sequence.shape[1])).astype(np.float32)
        scale = np.random.uniform(0.97, 1.03)
        sequence = sequence * scale + shift

        T = sequence.shape[0]
        n_mask = max(1, int(round(T * np.random.uniform(0.08, 0.12))))
        n_mask = min(n_mask, T)
        mask_idx = np.random.choice(T, size=n_mask, replace=False)
        sequence[mask_idx] = 0.0

        return sequence.astype(np.float32)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        sequence = np.load(path).astype(np.float32)
        label = self._label_from_path(path)

        if self.augment:
            sequence = self._apply_augmentation(sequence)

        T = sequence.shape[0]
        if T < self.max_frames:
            pad = np.zeros((self.max_frames - T, sequence.shape[1]), dtype=np.float32)
            sequence = np.concatenate([sequence, pad], axis=0)
            real_length = T
        elif T > self.max_frames:
            sequence = sequence[: self.max_frames]
            real_length = self.max_frames
        else:
            real_length = T

        return (
            torch.from_numpy(sequence),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(real_length, dtype=torch.long),
        )


def build_dataloaders(data_dir=DEFAULT_DATA_DIR, max_frames=60, batch_size=32,
                       val_ratio=0.2, seed=42, num_workers=0, augment_train=True):
    reference = LipReadingDataset(data_dir=data_dir, max_frames=max_frames)
    file_paths = reference.file_paths

    labels_by_path = {p: reference.label_map[_nfc(os.path.basename(os.path.dirname(p)))] for p in file_paths}
    paths_by_class = {}
    for p in file_paths:
        paths_by_class.setdefault(labels_by_path[p], []).append(p)

    rng = np.random.default_rng(seed)
    train_paths, val_paths = [], []
    for label, paths in paths_by_class.items():
        paths = list(paths)
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * val_ratio)))
        val_paths.extend(paths[:n_val])
        train_paths.extend(paths[n_val:])
    rng.shuffle(train_paths)
    rng.shuffle(val_paths)

    train_ds = LipReadingDataset(data_dir=data_dir, max_frames=max_frames,
                                  file_paths=train_paths, augment=augment_train)
    val_ds = LipReadingDataset(data_dir=data_dir, max_frames=max_frames,
                                file_paths=val_paths, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)

    return reference, train_loader, val_loader


if __name__ == "__main__":
    dataset, train_loader, val_loader = build_dataloaders()
    print(f"Clases ({len(dataset.class_names)}): {dataset.class_names}")
    print(f"Dimensión detectada por frame: {dataset.landmark_dim}")
    print(f"Total muestras: {len(dataset)} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    sequences, labels, lengths = next(iter(train_loader))
    print(f"Batch de ejemplo -> sequences: {sequences.shape}, labels: {labels.shape}, lengths: {lengths.shape}")
