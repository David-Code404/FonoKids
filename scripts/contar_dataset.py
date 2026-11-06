"""
contar_dataset.py
--------------------
Cuenta cuántas muestras tenés por palabra, en dos lugares:
    1. sesiones_continuas/<palabra>/clips/  -> videos grabados (crudos)
    2. dataset_pt/<palabra>/                -> ya procesados con MediaPipe

También detecta HUECOS en la numeración (ej. tenés hasta el _0500.avi pero
solo 450 archivos existen) -- eso indica que se borraron clips después de
grabados, algo que ya pasó en este proyecto y que un conteo simple no avisa.

Te sirve para llevar la cuenta de tu meta (ej. 500 por palabra) sin tener
que contar a mano.

Uso:
    python contar_dataset.py
"""
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sesiones_continuas")
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset_pt")

NUMERO_AL_FINAL = re.compile(r"_(\d{4})\.avi$", re.IGNORECASE)


def listar_avi(folder):
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith(".avi")]


def count_pt(folder):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(".pt")])


def max_numero_toma(archivos):
    """El número de toma más alto que aparece en los nombres de archivo
    (ej. de 'asqueroso_persona_0500.avi' saca 500), o 0 si no hay ninguno."""
    numeros = []
    for f in archivos:
        m = NUMERO_AL_FINAL.search(f)
        if m:
            numeros.append(int(m.group(1)))
    return max(numeros) if numeros else 0


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

    print(f"{'Palabra':<25} {'Existen':>8} {'Máx toma':>9} {'Procesados':>11}")
    print("-" * 70)

    total_clips, total_pt, total_faltantes = 0, 0, 0
    palabras_con_huecos = []
    for word in words:
        archivos = listar_avi(os.path.join(SESSIONS_DIR, word, "clips"))
        n_clips = len(archivos)
        maximo = max_numero_toma(archivos)
        n_pt = count_pt(os.path.join(DATASET_DIR, word))

        total_clips += n_clips
        total_pt += n_pt

        aviso = ""
        if maximo > n_clips:
            faltan = maximo - n_clips
            total_faltantes += faltan
            palabras_con_huecos.append((word, faltan))
            aviso = f"  <- faltan {faltan} (se borraron después de grabados)"

        print(f"{word:<25} {n_clips:>8} {maximo:>9} {n_pt:>11}{aviso}")

    print("-" * 70)
    print(f"{'TOTAL':<25} {total_clips:>8} {'':>9} {total_pt:>11}")
    print(f"\nPalabras distintas: {len(words)}")

    if palabras_con_huecos:
        print(f"\n[AVISO] {total_faltantes} clips en total desaparecieron después de "
              "grabados (hay huecos en la numeración), en estas palabras:")
        for word, faltan in palabras_con_huecos:
            print(f"  - {word}: faltan {faltan}")

    if total_pt < total_clips:
        pendientes = total_clips - total_pt
        print(f"\n[AVISO] Tenés {pendientes} clips grabados que todavía no tienen .pt "
              "(recorte de boca) generado.")


if __name__ == "__main__":
    main()
