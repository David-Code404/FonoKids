"""
verificar_dataset.py
-----------------------
Revisa cada .pt de dataset_pt/ y marca como sospechosos los clips que
probablemente estén mal:
    1. Detección de cara muy baja (MediaPipe no encontró la boca en la
       mayoría de los frames) -- señal de mala iluminación/encuadre.
    2. Duración mucho más corta que el resto de tomas de esa MISMA palabra
       -- señal de que la frase quedó cortada a la mitad (no se dijo
       completa).

Por defecto SOLO REPORTA, no borra nada. Para borrar de verdad los
archivos marcados como sospechosos, correlo con --borrar.

Uso:
    python verificar_dataset.py             # solo reporta
    python verificar_dataset.py --borrar    # además borra los .pt + .pkl sospechosos
"""
import os
import sys
import glob
import pickle

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset_pt")

MIN_DETECTION_RATE = 0.7   # menos de 70% de frames con cara detectada -> sospechoso
MIN_LENGTH_RATIO = 0.5     # menos de 50% de la duración mediana de esa palabra -> sospechoso (frase cortada)


def safe_torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def detection_rate(pkl_path):
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    landmarks = data.get("landmarks", [])
    if not landmarks:
        return None
    detected = sum(1 for lm in landmarks if lm is not None)
    return detected / len(landmarks)


def main():
    apply_delete = "--borrar" in sys.argv

    word_dirs = sorted(d for d in glob.glob(os.path.join(DATASET_DIR, "*")) if os.path.isdir(d))
    if not word_dirs:
        print(f"No hay carpetas de palabras en {DATASET_DIR}.")
        return

    all_flagged = []
    print(f"Umbral de detección de cara: >= {MIN_DETECTION_RATE*100:.0f}% de los frames")
    print(f"Umbral de duración: >= {MIN_LENGTH_RATIO*100:.0f}% de la mediana de esa palabra\n")

    for word_dir in word_dirs:
        word = os.path.basename(word_dir)
        pt_paths = sorted(glob.glob(os.path.join(word_dir, "*.pt")))
        if not pt_paths:
            continue

        lengths = {}
        rates = {}
        for pt_path in pt_paths:
            sample = safe_torch_load(pt_path)
            T = sample["sequence"].shape[0]
            lengths[pt_path] = T
            pkl_path = pt_path[:-3] + ".pkl"
            rates[pt_path] = detection_rate(pkl_path)

        median_len = float(np.median(list(lengths.values())))
        print(f"[{word}] {len(pt_paths)} clips | duración mediana: {median_len:.0f} frames")

        for pt_path in pt_paths:
            reasons = []
            rate = rates[pt_path]
            length = lengths[pt_path]

            if rate is not None and rate < MIN_DETECTION_RATE:
                reasons.append(f"deteccion de cara baja ({rate*100:.0f}%)")
            if median_len > 0 and length < median_len * MIN_LENGTH_RATIO:
                reasons.append(f"muy corto ({length} frames vs mediana {median_len:.0f} -- "
                                f"probablemente la frase quedo cortada)")

            if reasons:
                all_flagged.append((pt_path, reasons))
                print(f"  [SOSPECHOSO] {os.path.basename(pt_path)}: {'; '.join(reasons)}")

    print(f"\n=== Total sospechosos: {len(all_flagged)} de {sum(len(glob.glob(os.path.join(w,'*.pt'))) for w in word_dirs)} ===")

    if not all_flagged:
        print("Dataset limpio, no hay nada para borrar.")
        return

    if not apply_delete:
        print("\nEsto fue solo un reporte -- no se borró nada.")
        print("Si querés borrar de verdad estos archivos, corré:")
        print("    python verificar_dataset.py --borrar")
    else:
        for pt_path, _ in all_flagged:
            pkl_path = pt_path[:-3] + ".pkl"
            os.remove(pt_path)
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
        print(f"\nBorrados {len(all_flagged)} clips sospechosos (.pt + .pkl).")


if __name__ == "__main__":
    main()
