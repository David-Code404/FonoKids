"""
generar_sintetico.py
----------------------
Genera clips SINTÉTICOS de FRASES NEUTRAS (no bullying) para la clase
`sintetic`, directo en el espacio de landmarks -- sin grabar nada nuevo.

IMPORTANTE (v2): la primera versión armaba los clips pegando pedazos de
palabras de BULLYING entre sí -- eso estaba mal, el resultado no tenía
nada que ver con una frase neutra de verdad. Esta versión arma cada clip
como una secuencia de "visemas" (formas de boca) interpolados suavemente,
uno por sílaba de una frase neutra real (hola, qué tal, cómo estás, etc.)
-- conceptualmente más parecido a cómo se mueve la boca al decir esa
frase, aunque sin ser una síntesis fonética exacta.

Cómo se consiguen los visemas: no hay forma de "inventar" formas de boca
realistas de la nada, así que se agrupan (K-Means) todos los frames de
TODOS los clips de bullying ya grabados (misma cámara/backend que el resto
del dataset -- por eso no se usa el dataset externo CMU-MOSEAS, que viene
de otra cámara y no generalizó bien) en K formas de boca representativas
("visemas empíricos": boca cerrada, abierta, redondeada, etc., las que
aparecen de verdad al hablar frente a ESTA cámara). Son formas de boca
genéricas, no le pertenecen a ninguna palabra en particular.

Por cada frase neutra (ver FRASES_NEUTRAS abajo) se arma un clip:
    1. Un "visema objetivo" al azar por sílaba (evitando repetir el mismo
       visema dos sílabas seguidas, como en el habla real).
    2. Interpolación suave (coseno) entre visemas consecutivos a lo largo
       de los frames de esa sílaba -- así la boca no salta de golpe.
    3. Un poco de ruido/jitter por frame (coarticulación/jitter natural).
    4. Se recalcula suavizado + velocidad/aceleración desde cero, igual
       que un clip real (ver smooth_positions/add_dynamics).

Uso:
    python scripts/generar_sintetico.py [--por-frase 20] [--seed 42]

Requiere: haber corrido extraer_landmarks_npy.py antes (dataset base en
data/landmarks_npy/<palabra>/*.npy) -- de ahí saca los visemas empíricos.

Escribe en:
    data/landmarks_npy/<frase>/*.npy -- UNA CARPETA POR FRASE (hola/,
        gracias/, como_estas/, etc.), directo en data/landmarks_npy/, igual
        que las carpetas de las palabras de bullying -- NO todas juntas en
        una sola carpeta "sintetic".
    data/landmarks_npy/label_map.json (agrega las 20 frases como clases nuevas)
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import unicodedata

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extraer_landmarks_npy import smooth_positions, add_dynamics, OUT_DIR, LABEL_MAP_PATH  # noqa: E402

N_VISEMAS = 14            # cuántas formas de boca distintas agrupar con K-Means
FRAMES_POR_SILABA = (10, 18)   # cuántos frames dura cada sílaba (a ~25fps, ~0.4-0.7s)
JITTER_STD = 0.006        # ruido chico por frame, simula jitter real de detección

# Calibración contra el dataset real (medido sobre 5659 clips de bullying):
# |velocidad| media por frame = 0.00703, std = 0.00202. Los visemas salen
# de K-Means sobre TODOS los clips juntos (de personas/sesiones distintas)
# -- la diferencia de forma de boca ENTRE personas es mucho más grande que
# el movimiento real DENTRO de una sola frase, así que interpolar directo
# entre dos visemas de personas distintas da un movimiento gigante (~13x
# más rápido que el real). Por eso cada clip sintético se reescala DESPUÉS
# para que su velocidad quede en el rango real, antes de suavizar/derivar.
#
# OJO: se calibra por VELOCIDAD, no por "rango" (max-min) -- el rango mezcla
# las 40 dimensiones de posición, que tienen escalas base distintas entre
# sí (coordenadas de puntos de labios distintos), así que un reescalado
# alrededor del promedio POR DIMENSIÓN no controla el rango global de forma
# predecible. La velocidad (diferencia entre frames consecutivos) sí escala
# de forma exacta con cualquier reescalado alrededor de un centro fijo en
# el tiempo: escalar posiciones por `s` escala la velocidad por exactamente
# `s`, sin importar la diferencia de escala entre dimensiones.
# El objetivo se calibra ANTES del suavizado final (smooth_positions), que
# atenúa la velocidad de nuevo -- probado empíricamente: para que la
# velocidad DESPUÉS de suavizar termine en ~0.00703 (la real), hay que
# apuntar a ~1.96x eso antes de suavizar.
_POST_SMOOTH_ATTENUATION = 1.96
REAL_VEL_MEAN = 0.00703 * _POST_SMOOTH_ATTENUATION
REAL_VEL_STD = 0.00202 * _POST_SMOOTH_ATTENUATION
REAL_VEL_MIN = 0.003

# Frases neutras (NO de riesgo) típicas de una conversación normal, con su
# separación aproximada en sílabas -- no hace falta que sea foneticamente
# perfecta, solo una duración/cantidad de "golpes de boca" razonable.
FRASES_NEUTRAS = {
    "hola": ["ho", "la"],
    "que_tal": ["que", "tal"],
    "como_estas": ["co", "mo", "es", "tas"],
    "buenos_dias": ["bue", "nos", "di", "as"],
    "buenas_tardes": ["bue", "nas", "tar", "des"],
    "gracias": ["gra", "cias"],
    "por_favor": ["por", "fa", "vor"],
    "chau": ["chau"],
    "nos_vemos": ["nos", "ve", "mos"],
    "todo_bien": ["to", "do", "bien"],
    "que_hora_es": ["que", "ho", "ra", "es"],
    "nada_nuevo": ["na", "da", "nue", "vo"],
    "hasta_luego": ["has", "ta", "lue", "go"],
    "un_gusto": ["un", "gus", "to"],
    "muchas_gracias": ["mu", "chas", "gra", "cias"],
    "perdon": ["per", "don"],
    "disculpa": ["dis", "cul", "pa"],
    "buenas_noches": ["bue", "nas", "no", "ches"],
    "como_te_va": ["co", "mo", "te", "va"],
    "hasta_manana": ["has", "ta", "ma", "nia", "na"],
}

# No se usan como fuente para los visemas -- "no_es_riesgo" ya es habla
# normal (no bullying), y las carpetas de FRASES_NEUTRAS son las que este
# mismo script genera (si ya corrió antes, no hay que retroalimentarse de
# la propia data sintética).
EXCLUDE_AS_SOURCE = {"no_es_riesgo"} | set(FRASES_NEUTRAS.keys())


def _nfc(text):
    return unicodedata.normalize("NFC", text)


def load_real_frames(landmarks_dir):
    """Frames de POSICIÓN cruda (sin dinámica) de los clips de bullying ya
    grabados, CENTRADOS por clip (se resta el promedio temporal de CADA
    clip antes de juntar todo) -- así los visemas que salgan después
    representan la FORMA del movimiento (desviación respecto al propio
    reposo de esa persona/sesión), no la posición absoluta. Mezclar
    posiciones absolutas de gente/sesiones distintas daba visemas que, al
    combinarlos en un solo clip sintético, formaban una cara "imposible"
    (fue lo que hizo que la v1 se moviera 5x más de lo real).

    También devuelve la lista de centros (promedio por clip) reales, para
    poder anclar cada clip sintético a un centro real y consistente."""
    all_frames, centers = [], []
    for class_dir in sorted(glob.glob(os.path.join(landmarks_dir, "*"))):
        word = os.path.basename(class_dir)
        if not os.path.isdir(class_dir) or word in EXCLUDE_AS_SOURCE:
            continue
        for f in glob.glob(os.path.join(class_dir, "*.npy")):
            arr = np.load(f).astype(np.float32)
            position_dim = arr.shape[1] // 3
            positions = arr[:, :position_dim]
            center = positions.mean(axis=0, keepdims=True)
            all_frames.append(positions - center)  # solo la FORMA, sin la identidad
            centers.append(center[0])
    if not all_frames:
        return None, None
    return np.concatenate(all_frames, axis=0), np.stack(centers, axis=0)


def build_visemas(all_frames, n_visemas, seed):
    """K-Means sobre las DESVIACIONES reales (ya centradas por clip, ver
    load_real_frames) -- cada centroide es una forma de boca representativa
    ("visema empírico") relativa al reposo de una persona, no una posición
    absoluta -- así se pueden combinar visemas de clips distintos sin
    mezclar identidades."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_visemas, random_state=seed, n_init=10)
    km.fit(all_frames)
    return km.cluster_centers_.astype(np.float32)  # (n_visemas, position_dim)


def pick_phrase_signature(visemas, n_silabas, rng):
    """Elige la secuencia de visemas FIJA ("firma") de una frase -- se llama
    UNA sola vez por frase, no por clip. Antes se elegía una secuencia al
    azar en CADA clip generado, así que ni siquiera los 500 clips de UNA
    MISMA frase se parecían entre sí -- no había nada consistente que el
    modelo pudiera aprender a reconocer como "esto es más o menos hola".
    Con una firma fija por frase, todos sus clips comparten la misma
    secuencia de formas de boca (con variación de timing/ruido/identidad
    entre clips, ver make_phrase_clip), que es lo que hace que una frase
    sea reconocible como distinta de otra."""
    n_visemas = visemas.shape[0]
    targets = []
    last = -1
    for _ in range(n_silabas):
        choices = [i for i in range(n_visemas) if i != last]
        idx = rng.choice(choices)
        targets.append(idx)
        last = idx
    return targets


def make_phrase_clip(visemas, real_centers, targets, rng):
    deviations = []
    prev_vec = visemas[targets[0]]
    for target_idx in targets:
        n_frames = rng.randint(*FRAMES_POR_SILABA)
        target_vec = visemas[target_idx]
        for t in range(n_frames):
            # Interpolación coseno (suave al principio/final del tramo, no lineal a los tumbos).
            alpha = 0.5 - 0.5 * math.cos(math.pi * (t + 1) / n_frames)
            vec = (1 - alpha) * prev_vec + alpha * target_vec
            vec = vec + rng.gauss(0, 1) * JITTER_STD  # jitter escalar simple, barato y suficiente
            deviations.append(vec.copy())
        prev_vec = target_vec

    deviations = np.stack(deviations, axis=0).astype(np.float32)  # (T, position_dim), promedio ~0

    # Reescalado a una velocidad de movimiento realista (ver REAL_VEL_MEAN
    # más arriba) -- las desviaciones ya están centradas en cero por
    # construcción (vienen de visemas centrados por clip), así que se
    # escalan directo, sin restar/sumar ningún centro acá.
    current_vel = float(np.abs(np.diff(deviations, axis=0)).mean()) if len(deviations) > 1 else 0.0
    target_vel = max(REAL_VEL_MIN, rng.gauss(REAL_VEL_MEAN, REAL_VEL_STD))
    scale = target_vel / max(current_vel, 1e-6)
    deviations = deviations * scale

    # Se ancla el clip a la "cara base" de UNA sola persona/sesión real
    # (elegida al azar entre los centros reales) -- así el resultado
    # entero corresponde a una identidad consistente, en vez de mezclar
    # las proporciones de varias personas distintas.
    base_center = real_centers[rng.randrange(len(real_centers))]
    positions = base_center[np.newaxis, :] + deviations

    smoothed = smooth_positions(positions)
    sequence = add_dynamics(smoothed)  # (T, LANDMARK_DIM)
    return sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--por-frase", type=int, default=20,
                         help="cuántas variantes sintéticas generar POR frase neutra "
                              f"(default 20 -- con las {len(FRASES_NEUTRAS)} frases de "
                              "FRASES_NEUTRAS da ~280 clips en total)")
    parser.add_argument("--visemas", type=int, default=N_VISEMAS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(LABEL_MAP_PATH):
        print(f"No encontré {LABEL_MAP_PATH}. Corré extraer_landmarks_npy.py al menos una vez antes.")
        return

    print("Juntando frames reales para calcular visemas empíricos...")
    all_frames, real_centers = load_real_frames(OUT_DIR)
    if all_frames is None:
        print(f"No encontré clips de bullying en {OUT_DIR} para sacar visemas.")
        return
    print(f"  {all_frames.shape[0]} frames de {all_frames.shape[1]} dims "
          f"({len(real_centers)} clips/identidades) -- corriendo K-Means (k={args.visemas})...")
    visemas = build_visemas(all_frames, args.visemas, args.seed)
    print(f"  {len(visemas)} visemas empíricos listos.")

    rng = random.Random(args.seed)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    # Eliminamos la clase vieja "sintetic" (todas las frases juntas en una
    # sola carpeta) si quedó de una corrida anterior -- ahora cada frase es
    # su propia clase, directo en data/landmarks_npy/, igual que las
    # palabras de bullying.
    if "sintetic" in label_map:
        del label_map["sintetic"]
    for frase in FRASES_NEUTRAS:
        if frase not in label_map:
            label_map[frase] = len(label_map)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    total = 0
    firmas_usadas = set()  # evita que dos frases con la misma cantidad de sílabas terminen con la MISMA firma
    for frase, silabas in FRASES_NEUTRAS.items():
        # Firma FIJA de la frase (ver pick_phrase_signature) -- se elige UNA
        # vez acá y se reusa para los N clips de esta frase, así todos
        # comparten la misma secuencia de formas de boca (identidad
        # reconocible), variando solo timing/ruido/identidad de base.
        targets = pick_phrase_signature(visemas, len(silabas), rng)
        intentos = 0
        while tuple(targets) in firmas_usadas and intentos < 50:
            targets = pick_phrase_signature(visemas, len(silabas), rng)
            intentos += 1
        firmas_usadas.add(tuple(targets))

        out_dir = os.path.join(OUT_DIR, frase)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(args.por_frase):
            sequence = make_phrase_clip(visemas, real_centers, targets, rng)
            np.save(os.path.join(out_dir, f"{frase}_{i:03d}.npy"), sequence)
            total += 1
        print(f"  {frase:16s} (label {label_map[frase]:2d}, firma={targets}) -> "
              f"{args.por_frase} clips en {out_dir}")

    print(f"\n=== TOTAL: {total} clips sintéticos de {len(FRASES_NEUTRAS)} frases neutras, "
          f"cada una en su propia carpeta dentro de {OUT_DIR} ===")
    print("Ahora podés reentrenar con: python scripts/train_landmarks_transformer.py")


if __name__ == "__main__":
    main()
