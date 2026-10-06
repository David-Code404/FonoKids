"""
merge_dataset_externo.py
-------------------------
Mezcla un dataset externo (landmarks_npy + audio_embeddings_cache ya
extraídos, ej. bajado de Drive por un compañero) con el dataset local,
renumerando los archivos para no pisar nada -- cada archivo nuevo sigue
la numeración donde la dejó el dataset local (ej. si "casa_correcto" tiene
hasta el 0201, lo nuevo arranca en 0202).

Uso:
    python scripts/merge_dataset_externo.py "<ruta a la carpeta data/ externa>"

No toca el dataset local existente, solo AGREGA archivos nuevos con
nombres nuevos -- se puede correr de nuevo sin duplicar si se le pasa la
misma carpeta dos veces (usa el conteo real de archivos destino, no un
contador en memoria).
"""
import os
import re
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def siguiente_indice(carpeta_destino, prefijo_clase):
    """Más alto índice "_NNNN" ya usado en carpeta_destino para esa clase,
    +1 -- 1 si todavía no hay ninguno."""
    maximo = 0
    if os.path.isdir(carpeta_destino):
        for f in os.listdir(carpeta_destino):
            m = re.match(rf"^{re.escape(prefijo_clase)}_(\d+)\.npy$", f)
            if m:
                maximo = max(maximo, int(m.group(1)))
    return maximo + 1


def merge_landmarks(origen_data, destino_data):
    origen_dir = os.path.join(origen_data, "landmarks_npy")
    destino_dir = os.path.join(destino_data, "landmarks_npy")
    if not os.path.isdir(origen_dir):
        print("[landmarks] no hay landmarks_npy en el origen, se salta.")
        return

    for clase in sorted(os.listdir(origen_dir)):
        carpeta_origen_clase = os.path.join(origen_dir, clase)
        if not os.path.isdir(carpeta_origen_clase):
            continue
        carpeta_destino_clase = os.path.join(destino_dir, clase)
        os.makedirs(carpeta_destino_clase, exist_ok=True)

        archivos = sorted(f for f in os.listdir(carpeta_origen_clase) if f.endswith(".npy"))
        idx = siguiente_indice(carpeta_destino_clase, clase)
        copiados = 0
        for f in archivos:
            nombre_nuevo = f"{clase}_{idx:04d}.npy"
            shutil.copy2(
                os.path.join(carpeta_origen_clase, f),
                os.path.join(carpeta_destino_clase, nombre_nuevo),
            )
            idx += 1
            copiados += 1
        print(f"[landmarks] {clase}: +{copiados} (quedó con {len(os.listdir(carpeta_destino_clase))})")


def merge_audio_cache(origen_data, destino_data):
    origen_dir = os.path.join(origen_data, "audio_embeddings_cache")
    destino_dir = os.path.join(destino_data, "audio_embeddings_cache")
    if not os.path.isdir(origen_dir):
        print("[audio] no hay audio_embeddings_cache en el origen, se salta.")
        return
    os.makedirs(destino_dir, exist_ok=True)

    por_clase = {}
    for f in sorted(os.listdir(origen_dir)):
        if not f.endswith(".npy"):
            continue
        clase = re.sub(r"_\d+$", "", os.path.splitext(f)[0])
        por_clase.setdefault(clase, []).append(f)

    for clase, archivos in sorted(por_clase.items()):
        idx = siguiente_indice(destino_dir, clase)
        copiados = 0
        for f in archivos:
            nombre_nuevo = f"{clase}_{idx:04d}.npy"
            shutil.copy2(os.path.join(origen_dir, f), os.path.join(destino_dir, nombre_nuevo))
            idx += 1
            copiados += 1
        total = len([x for x in os.listdir(destino_dir) if x.startswith(clase + "_")])
        print(f"[audio] {clase}: +{copiados} (quedó con {total})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python scripts/merge_dataset_externo.py <ruta a la carpeta data/ externa>")
        sys.exit(1)

    origen_data = sys.argv[1]
    destino_data = os.path.join(BASE_DIR, "data")

    if not os.path.isdir(origen_data):
        print(f"No existe la carpeta: {origen_data}")
        sys.exit(1)

    print(f"Origen:  {origen_data}")
    print(f"Destino: {destino_data}\n")

    merge_landmarks(origen_data, destino_data)
    print()
    merge_audio_cache(origen_data, destino_data)
    print("\nListo.")
