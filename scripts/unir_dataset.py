"""
unir_dataset.py
-------------------
Junta clips grabados en OTRA PC (por ejemplo, copiaste su carpeta
sesiones_continuas a un pendrive) con tu dataset local, sin pisar nada:
renumera los clips que vienen de afuera para que no choquen con los que ya
tenés (ambas PCs arrancan la numeración en 0001, así que copiarlos directo
pisaría clips existentes).

Es NO DESTRUCTIVO: copia los archivos (no los mueve, no borra nada), así que
podés correrlo las veces que haga falta.

CÓMO USARLO:
    1. Copiá la carpeta "sesiones_continuas" de la otra PC a cualquier lugar
       de esta (un pendrive, el escritorio, donde sea).
    2. Cambiá la variable RUTA_ORIGEN de más abajo para que apunte a esa
       carpeta copiada.
    3. Corré:  python unir_dataset.py

Después de unir, corré extraer_landmarks_npy.py de nuevo para que los
landmarks incluyan también los clips nuevos.
"""
import os
import shutil

# =====================================================================
# CAMBIÁ ESTA RUTA por donde copiaste los datos de la otra PC.
# Tiene que ser una carpeta que tenga subcarpetas de palabras adentro,
# cada una con su "clips" (la misma forma que sesiones_continuas):
#   RUTA_ORIGEN/asqueroso/clips/asqueroso_0001.avi, ...
#   RUTA_ORIGEN/camba_de_mierda/clips/camba_de_mierda_0001.avi, ...
# =====================================================================
RUTA_ORIGEN = r"C:\Users\davit\Downloads\asqueroso-20260902T190243Z-1-001"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTINO = os.path.join(BASE_DIR, "data", "D:\SpeakShadow\prueba")


def contar_clips_existentes(clips_dir):
    if not os.path.isdir(clips_dir):
        return 0
    return len([f for f in os.listdir(clips_dir) if f.lower().endswith(".avi")])


def sacar_persona(word, filename):
    """De '<palabra>_<persona>_0001.avi' saca la persona, o None si es un
    clip viejo sin persona en el nombre ('<palabra>_0001.avi')."""
    stem = filename[:-4]  # sin ".avi"
    prefix = word + "_"
    rest = stem[len(prefix):] if stem.lower().startswith(prefix.lower()) else stem
    if "_" in rest:
        persona, _seq = rest.rsplit("_", 1)
        return persona
    return None


def unir_palabra(word, origen_clips, destino_clips):
    os.makedirs(destino_clips, exist_ok=True)
    siguiente = contar_clips_existentes(destino_clips) + 1

    archivos = sorted(f for f in os.listdir(origen_clips) if f.lower().endswith(".avi"))
    if not archivos:
        return 0

    copiados = 0
    for filename in archivos:
        persona = sacar_persona(word, filename)
        sufijo = f"{persona}_" if persona else ""
        nuevo_nombre = f"{word}_{sufijo}{siguiente:04d}.avi"

        origen_path = os.path.join(origen_clips, filename)
        destino_path = os.path.join(destino_clips, nuevo_nombre)
        shutil.copy2(origen_path, destino_path)

        siguiente += 1
        copiados += 1

    return copiados


def main():
    if not os.path.isdir(RUTA_ORIGEN):
        print(f"No encontré la carpeta de origen: {RUTA_ORIGEN}")
        print("Editá la variable RUTA_ORIGEN en este archivo con la ruta correcta.")
        return

    palabras = sorted(
        d for d in os.listdir(RUTA_ORIGEN) if os.path.isdir(os.path.join(RUTA_ORIGEN, d))
    )
    if not palabras:
        print(f"No hay ninguna carpeta de palabra dentro de {RUTA_ORIGEN}.")
        return

    print(f"Uniendo {len(palabras)} palabras desde:\n  {RUTA_ORIGEN}\nhacia:\n  {DESTINO}\n")

    total = 0
    for word in palabras:
        origen_clips = os.path.join(RUTA_ORIGEN, word, "clips")
        if not os.path.isdir(origen_clips):
            print(f"[{word}] no tiene carpeta 'clips', se omite.")
            continue

        destino_clips = os.path.join(DESTINO, word, "clips")
        copiados = unir_palabra(word, origen_clips, destino_clips)
        total += copiados
        print(f"[{word}] {copiados} clips copiados -> {destino_clips}")

    print(f"\n=== TOTAL: {total} clips unidos ===")
    print("Ahora corré: python extraer_landmarks_npy.py  (para procesar los clips nuevos)")


if __name__ == "__main__":
    main()
