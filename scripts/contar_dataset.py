"""
contar_dataset.py
--------------------
Cuenta cuántas muestras tenés por palabra, en dos lugares:
    1. sesiones_continuas/<palabra>/clips/  -> videos grabados (crudos)
    2. dataset_pt/<palabra>/                -> ya procesados con MediaPipe

Te sirve para llevar la cuenta de tu meta (ej. 500 por palabra) sin tener
que contar a mano.

Uso:
    python contar_dataset.py
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset_pt")


def count_avi(folder):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(".avi")])


def count_pt(folder):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(".pt")])


def main():
    words = set()
    if os.path.isdir(SESSIONS_DIR):
        words.update(d for d in os.listdir(SESSIONS_DIR)
                     if os.path.isdir(os.path.join(SESSIONS_DIR, d)))
    if os.path.isdir(DATASET_DIR):
        words.update(d for d in os.listdir(DATASET_DIR)
                     if os.path.isdir(os.path.join(DATASET_DIR, d)))

    if not words:
        print("No hay ninguna palabra grabada todavía.")
        return

    words = sorted(words)

    print(f"{'Palabra':<25} {'Grabados (clips)':>18} {'Procesados (dataset_pt)':>25}")
    print("-" * 70)

    total_clips, total_pt = 0, 0
    for word in words:
        n_clips = count_avi(os.path.join(SESSIONS_DIR, word, "clips"))
        n_pt = count_pt(os.path.join(DATASET_DIR, word))
        total_clips += n_clips
        total_pt += n_pt
        print(f"{word:<25} {n_clips:>18} {n_pt:>25}")

    print("-" * 70)
    print(f"{'TOTAL':<25} {total_clips:>18} {total_pt:>25}")
    print(f"\nPalabras distintas: {len(words)}")

    if total_pt < total_clips:
        pendientes = total_clips - total_pt
        print(f"\n[AVISO] Tenés {pendientes} clips grabados que todavía no tienen .pt "
              "(recorte de boca) generado.")


if __name__ == "__main__":
    main()
