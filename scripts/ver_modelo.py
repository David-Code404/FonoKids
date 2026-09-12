"""
ver_modelo.py
--------------
Muestra qué hay adentro de models/mejor_modelo_landmarks_conformer.pth --
sobre todo, qué palabras/frases conoce (class_names). Lee el checkpoint
directo, así que es automático: si mañana el modelo tiene más o menos
clases, esto las muestra igual, sin tocar el código.

Uso:
    python scripts/ver_modelo.py
"""
import os

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mejor_modelo_landmarks_conformer.pth")


def safe_torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"No encontré {MODEL_PATH}.")
        return

    checkpoint = safe_torch_load(MODEL_PATH)
    class_names = checkpoint.get("class_names", [])
    max_frames = checkpoint.get("max_frames")
    input_dim = checkpoint.get("input_dim")
    val_acc = checkpoint.get("val_acc")
    val_loss = checkpoint.get("val_loss")

    print(f"Archivo: {MODEL_PATH}")
    print(f"Tamaño: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB")
    print(f"Modificado: {os.path.getmtime(MODEL_PATH)}")
    print()
    print(f"Cantidad de clases: {len(class_names)}")
    print(f"max_frames: {max_frames}  |  input_dim: {input_dim}")
    if val_acc is not None:
        print(f"Val Acc guardado: {val_acc:.2f}%")
    if val_loss is not None:
        print(f"Val Loss guardado: {val_loss:.4f}")
    print()
    print("Clases (en orden, índice = label):")
    for i, name in enumerate(class_names):
        print(f"  {i:3d}  {name}")


if __name__ == "__main__":
    main()
