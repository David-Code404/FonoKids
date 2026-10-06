"""
audio_pronunciation.py
-----------------------
Evalúa la pronunciación de una palabra a partir del AUDIO de un clip (el
.wav que graba scripts/grabar_video_continuo.py junto al .avi), usando el
modelo pre-entrenado facebook/wav2vec2-xlsr-53-espeak-cv-ft.

Este modelo no transcribe palabras -- transcribe FONEMAS (símbolos IPA),
que es justo lo que hace falta acá: para decirle a un chico "el problema
estuvo en la erre", hay que saber qué fonema salió mal, no solo si la
palabra en general se entendió.

Cómo funciona (GOP -- Goodness of Pronunciation, simplificado):
    1. Se convierte la palabra objetivo a su secuencia de fonemas esperada
       (ver texto_a_fonemas() más abajo -- reglas de español, no depende de
       espeak-ng porque ese binario no viene instalado en Windows).
    2. El audio pasa por el modelo, que da una probabilidad por fonema en
       CADA frame de audio (~50 frames por segundo).
    3. torchaudio.functional.forced_align (alineación forzada por CTC) ubica
       CADA fonema esperado en su tramo de frames dentro del audio real.
    4. El puntaje de cada fonema es el promedio de probabilidad que el
       modelo le dio en su tramo -- alto = lo pronunció claro, bajo = el
       modelo dudó mucho ahí (probable error de pronunciación).

Uso (standalone, para probar con un .wav ya grabado):
    python audio_pronunciation.py <ruta_al_wav> <palabra>

Ejemplo:
    python audio_pronunciation.py ../data/sesiones_continuas/perro_correcto/clips/perro_correcto_0001.wav perro
"""
import sys

# La consola de Windows por defecto usa cp1252, que no tiene la mayoría de
# los símbolos IPA (ɾ, ʝ, ɲ, etc.) -- sin esto, print() tira
# UnicodeEncodeError o muestra "?" en vez del fonema real. reconfigure()
# existe desde Python 3.7.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
TARGET_SAMPLE_RATE = 16000

# Umbral de log-probabilidad promedio por fonema: por debajo de esto, se
# marca ese fonema puntual como "sospechoso" en el reporte. Valor elegido a
# ojo (no calibrado contra un dataset real todavía) -- ajustar cuando haya
# grabaciones reales de sobra para ver la distribución típica de scores.
GOP_SUSPICIOUS_THRESHOLD = -3.0

# forced_align OBLIGA al audio a encajar contra los fonemas esperados, pase
# lo que pase -- si decís "pedo" cuando la palabra era "perro", igual arma
# una alineación (mala, pero alineación), así que el GOP por sí solo no
# siempre detecta que dijiste OTRA palabra. Por eso se compara además el
# reconocimiento LIBRE (sin forzar nada) contra lo esperado, con distancia
# de edición normalizada -- por encima de este umbral, se considera que no
# dijo la palabra pedida, más allá de cómo haya salido el GOP fonema por
# fonema. 0.5 = hasta la mitad de los fonemas pueden diferir antes de
# marcarlo (tolera errores de pronunciación reales, no exige match exacto).
PALABRA_MISMATCH_THRESHOLD = 0.5


# =====================================================================
# Texto -> fonemas (reglas de español, sin depender de espeak-ng)
# =====================================================================
# Simplificaciones a propósito, documentadas para no esconder el error que
# introducen:
#   - No se marca acento/stress -- el modelo de fonemas no lo necesita para
#     esta tarea (solo importa el sonido, no qué sílaba se enfatiza).
#   - Diptongos (ej. "ió" en "camión") se parten en vocales sueltas en vez
#     de usar semivocales (j/w) -- una aproximación razonable para palabras
#     cortas de una sílaba o dos, puede perder precisión en diptongos raros.
#   - Español latinoamericano (seseo: c/z -> s: no se distingue de la "z"
#     castellana; yeísmo: ll/y -> ʝ) -- coincide con el español que ya usa
#     el resto del dataset del proyecto (dialecto boliviano).
VOCALES_ACENTUADAS = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u"}


def texto_a_fonemas(palabra):
    """'perro' -> ['p','e','r','o']. 'carro' -> ['k','a','r','o'].
    'flor' -> ['f','l','o','ɾ'] (r suave, no vibrante, porque no está al
    principio de palabra ni duplicada)."""
    palabra = palabra.strip().lower()
    for acentuada, simple in VOCALES_ACENTUADAS.items():
        palabra = palabra.replace(acentuada, simple)

    fonemas = []
    i = 0
    n = len(palabra)
    while i < n:
        c = palabra[i]
        nxt = palabra[i + 1] if i + 1 < n else ""
        prev = palabra[i - 1] if i > 0 else ""

        if c == "r":
            doble = nxt == "r"
            al_inicio = i == 0
            tras_nls = prev in ("n", "l", "s")
            fonemas.append("r" if (doble or al_inicio or tras_nls) else "ɾ")
            i += 2 if doble else 1
            continue

        if c == "c" and nxt in ("e", "i"):
            fonemas.append("s")  # seseo
            i += 1
            continue
        if c == "c" and nxt == "h":
            fonemas.append("tʃ")
            i += 2
            continue
        if c == "c":
            fonemas.append("k")
            i += 1
            continue

        if c == "q" and nxt == "u":
            fonemas.append("k")
            i += 2  # la "u" de "qu" es muda
            continue

        if c == "g" and nxt == "u" and i + 2 < n and palabra[i + 2] in ("e", "i"):
            fonemas.append("ɡ")
            i += 2  # "gu" antes de e/i -> "u" muda
            continue
        if c == "g" and nxt in ("e", "i"):
            fonemas.append("x")
            i += 1
            continue
        if c == "g":
            fonemas.append("ɡ")
            i += 1
            continue

        if c == "l" and nxt == "l":
            fonemas.append("ʝ")  # yeísmo
            i += 2
            continue

        if c == "y":
            # Vocal (al final de palabra, tras vocal, ej. "hoy") vs
            # consonante (ej. "yo", yeísmo -- mismo sonido que "ll").
            es_vocal_final = (i == n - 1) and prev in "aeiou"
            fonemas.append("i" if es_vocal_final else "ʝ")
            i += 1
            continue

        if c == "z":
            fonemas.append("s")  # seseo
            i += 1
            continue

        if c == "j":
            fonemas.append("x")
            i += 1
            continue

        if c == "h":
            i += 1  # muda
            continue

        if c == "x":
            fonemas.extend(["k", "s"])
            i += 1
            continue

        if c == "v":
            fonemas.append("b")
            i += 1
            continue

        if c == "ñ":
            fonemas.append("ɲ")
            i += 1
            continue

        if c in "aeiou":
            fonemas.append(c)
            i += 1
            continue

        if c in "ptdkbfmnlsw":
            fonemas.append(c)
            i += 1
            continue

        # Símbolo no reconocido (espacio, guion, etc.) -- se ignora en vez
        # de romper la conversión de toda la palabra.
        i += 1

    return fonemas


# =====================================================================
# Modelo (se carga UNA sola vez -- ver load_model())
# =====================================================================
# OJO: a propósito NO se usa Wav2Vec2Processor.from_pretrained() acá -- ese
# processor instancia por dentro un Wav2Vec2PhonemeCTCTokenizer, que en su
# __init__ exige tener el binario `espeak-ng` instalado en el sistema
# (no alcanza con `pip install phonemizer`, que es solo el wrapper Python)
# incluso si nunca se usa su función de convertir texto a fonemas -- que acá
# no hace falta, texto_a_fonemas() ya hace ese trabajo. En Windows ese
# binario no viene con pip, así que se evita todo ese camino: se carga el
# extractor de audio (Wav2Vec2FeatureExtractor, sin esa dependencia) y el
# vocabulario de fonemas se lee directo del vocab.json del repo.
def load_model(device=None):
    import json
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando {MODEL_NAME} (primera vez tarda, baja ~1.2GB)...")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    # OJO: a propósito NO se usa Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    # -- ese checkpoint solo existe en el repo como pytorch_model.bin (sin
    # versión safetensors), y transformers >=4.5x bloquea directamente
    # cargar .bin con torch.load si torch < 2.6 (CVE-2025-32434), aunque el
    # torch instalado (2.5.1, el que usa el resto del proyecto para el
    # pipeline visual con GPU) SÍ soporta torch.load(weights_only=True) de
    # forma segura -- es transformers el que lo bloquea igual, no torch. En
    # vez de forzar una actualización de torch que podría romper la
    # compatibilidad de CUDA para el modelo visual, se arma el modelo desde
    # su config y se cargan los pesos a mano, con weights_only=True (evita
    # ejecutar código arbitrario del pickle, que es el riesgo real del CVE).
    config = Wav2Vec2Config.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC(config)
    weights_path = hf_hub_download(MODEL_NAME, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    vocab_path = hf_hub_download(MODEL_NAME, "vocab.json")
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    print(f"Listo. Device: {device}. Vocabulario: {len(vocab)} fonemas.")
    return model, feature_extractor, vocab, device


def load_audio_16k(wav_path):
    """Devuelve un tensor 1D float32 mono a 16kHz, sea cual sea el
    sample rate/canales originales del .wav grabado."""
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # a mono
    if sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SAMPLE_RATE)
    return waveform.squeeze(0)


# Dimensión del embedding que devuelve extract_embedding() -- depende del
# modelo (wav2vec2-xlsr-53 grande = 1024). La usan train_audio_classifier.py
# (para armar la capa de entrada) y server/main.py (para validar que el
# checkpoint guardado coincide con el modelo cargado en runtime).
EMBEDDING_DIM = 1024


def extract_embedding(wav_path, model, feature_extractor, device):
    """Vector fijo (EMBEDDING_DIM,) que resume el AUDIO completo del clip --
    a diferencia de score_pronunciation() (que da un puntaje por fonema
    esperado), esto es lo que usa el clasificador entrenado con datos reales
    (train_audio_classifier.py) para decidir directamente "correcto" vs
    cada tipo de error, igual que el modelo visual hace con landmarks.

    Es el promedio en el tiempo del último estado oculto del encoder wav2vec2
    (antes de la cabeza CTC de fonemas) -- una representación acústica
    general del clip, no atada a ninguna palabra esperada."""
    waveform = load_audio_16k(wav_path)
    inputs = feature_extractor(waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        # model.wav2vec2 es el encoder base (sin la cabeza de fonemas) --
        # Wav2Vec2ForCTC lo expone como submódulo con ese nombre.
        hidden = model.wav2vec2(inputs.input_values.to(device)).last_hidden_state  # (1, T, 1024)
        pooled = hidden.mean(dim=1).squeeze(0)  # (1024,)
    return pooled.cpu().numpy()


def _phoneme_to_token_ids(fonemas, vocab):
    """Mapea cada símbolo IPA de texto_a_fonemas() a su id en el vocabulario
    del modelo. Si algún fonema no está en el vocabulario (no debería pasar
    con los símbolos que usa texto_a_fonemas(), todos estándar), se avisa y
    se saltea -- mejor perder un fonema puntual que romper todo el score."""
    ids = []
    fonemas_usados = []
    for f in fonemas:
        if f in vocab:
            ids.append(vocab[f])
            fonemas_usados.append(f)
        else:
            print(f"[AVISO] Fonema '{f}' no está en el vocabulario del modelo, se saltea.")
    return ids, fonemas_usados


def _greedy_ctc_decode(pred_ids, id_to_phoneme, blank_id):
    """Decodificación CTC libre (sin forzar ninguna palabra) -- colapsa
    repeticiones consecutivas y saca el blank, igual que hace
    processor.batch_decode() normalmente. Devuelve la lista de símbolos
    (no un string) porque además de mostrarse en el reporte, se usa para
    comparar contra los fonemas esperados (ver _distancia_edicion)."""
    fonemas = []
    anterior = None
    for tid in pred_ids:
        if tid != anterior and tid != blank_id:
            fonemas.append(id_to_phoneme.get(tid, "?"))
        anterior = tid
    return fonemas


# Inverso aproximado de texto_a_fonemas() -- para mostrarle al chico/padre
# QUÉ sonó, no solo "está mal". No es una transcripción fonética seria (evo
# ese título no hace falta acá), es nada más un texto legible parecido a como
# se escribiría en español lo que el modelo reconoció libremente.
_FONEMA_A_LETRA = {"r": "rr", "ɾ": "r", "x": "j", "ɡ": "g", "tʃ": "ch", "ʝ": "ll", "ɲ": "ñ"}


def fonemas_a_texto_aproximado(fonemas):
    """['p','e','l','o'] -> 'pelo'. Solo para mostrar, no para comparar."""
    return "".join(_FONEMA_A_LETRA.get(f, f) for f in fonemas)


def _distancia_edicion(a, b):
    """Distancia de Levenshtein entre dos listas de símbolos -- cuántas
    inserciones/borrados/sustituciones hacen falta para convertir `a` en
    `b`. Programación dinámica clásica, O(len(a)*len(b))."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


# =====================================================================
# Diagnóstico de tipo de error (omisión / lambdacismo / dentalización) a
# partir del audio -- sin depender de clases entrenadas por subtipo (el
# dataset real solo separa correcto/incorrecto, mezclado). Se alinea la
# secuencia de fonemas ESPERADA contra lo que el modelo reconoció LIBRE
# (sin forzar ninguna palabra) con distancia de edición clásica, y se mira
# qué pasó puntualmente en la posición del fonema "difícil" (la r
# vibrante/simple, que es lo único que el proyecto entrena hoy): si
# desapareció = omisión, si se convirtió en "l" = lambdacismo, si se
# convirtió en "d" = dentalización. Cualquier otra sustitución queda sin
# clasificar (None) en vez de inventar una etiqueta que no corresponde.
# =====================================================================
def _align_ops(a, b):
    """Alineación por distancia de Levenshtein con backtrace -- devuelve la
    lista de operaciones (match/sub/del/ins) en el orden de `a`, cada una
    con el símbolo de `a` (o None si fue insertado) y el símbolo de `b`
    alineado (o None si fue borrado)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            ops.append(("match", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", a[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, b[j - 1]))
            j -= 1
    ops.reverse()
    return ops


# Fonemas "difíciles" que el proyecto entrena hoy (ver web/src/soundCategories.js)
# y, para cada uno, qué sustituciones puntuales corresponden a qué nombre de
# error clínico -- mismo criterio que ya se usaba para R/RR, generalizado a
# las otras 4 categorías (C/K, G suave, S, Sinfones). Los sinfones (bl/cl/
# fl/gl/pl) se resuelven aparte, más abajo, porque el "error" ahí es sobre
# el SEGUNDO fonema del grupo (la "l"), no sobre uno solo.
_SUSTITUCIONES_CONOCIDAS = {
    "r": {"l": "lambdacismo", "d": "dentalizacion"},
    "ɾ": {"l": "lambdacismo", "d": "dentalizacion"},
    "k": {"t": "anteriorizacion", "x": "debilitamiento", "h": "debilitamiento"},
    "ɡ": {"d": "anteriorizacion", "x": "debilitamiento", "h": "debilitamiento"},
    "s": {"θ": "interdental", "t": "oclusion", "d": "oclusion"},
    "ʝ": {"l": "distorsion", "i": "debilitamiento"},
    "tʃ": {"ʃ": "distorsion", "t": "debilitamiento", "s": "debilitamiento"},
}
_SINFONES = {"bl", "kl", "fl", "ɡl", "pl"}  # cl -> "k"+"l" en fonemas (texto_a_fonemas usa "k")


def diagnose_tipo_error(fonemas_esperados, reconocido_libre_lista):
    """Devuelve un tipo de error (omision / lambdacismo / dentalizacion /
    anteriorizacion / debilitamiento / interdental / oclusion / epentesis)
    o None, mirando qué pasó en la posición del fonema difícil de la
    palabra dentro de la alineación entre lo esperado y lo reconocido
    libremente. Cubre los 5 sonidos que el proyecto entrena hoy (R/RR, S,
    G, C/K, Sinfones con L) -- si la palabra no tiene ninguno de esos
    fonemas, o el error no encaja en ningún patrón conocido, devuelve None
    en vez de forzar una etiqueta."""
    ops = _align_ops(fonemas_esperados, reconocido_libre_lista)

    # Sinfones (bl/cl/fl/gl/pl): el error típico es comerse la "l" del grupo
    # (reducción) o meter una vocal en el medio (epéntesis) -- se busca una
    # "l" esperada que venga justo después de una consonante de sinfón.
    for idx, (tipo, esperado, reconocido) in enumerate(ops):
        if esperado == "l" and idx > 0:
            anterior = ops[idx - 1][1]  # fonema esperado anterior
            if anterior and (anterior + "l") in _SINFONES:
                if tipo == "del":
                    return "omision"
                if tipo == "sub" and reconocido and reconocido not in ("l",):
                    return "distorsion"
    for idx, (tipo, esperado, reconocido) in enumerate(ops):
        if tipo == "ins" and idx > 0:
            previo = ops[idx - 1]
            if previo[0] in ("match", "sub") and previo[1] and (previo[1] + "l") in _SINFONES:
                return "epentesis"

    # R/RR, S, G, C/K: un solo fonema difícil, sustitución directa.
    for tipo, esperado, reconocido in ops:
        tabla = _SUSTITUCIONES_CONOCIDAS.get(esperado)
        if tabla is None:
            continue
        if tipo == "del":
            return "omision"
        if tipo == "sub" and reconocido in tabla:
            return tabla[reconocido]
    return None


def score_pronunciation(wav_path, palabra, model, feature_extractor, vocab, device):
    """Puntúa la pronunciación de `palabra` en el clip `wav_path`.

    Devuelve un dict:
        {
            "fonemas": ["p", "e", "r", "o"],
            "scores": [-0.12, -0.34, -2.81, -0.09],   # log-prob promedio, uno por fonema
            "sospechosos": [2],                        # índices por debajo del umbral
            "reconocido_libre": "p e ɾ o",              # qué reconoció el modelo SIN forzar la palabra
        }
    """
    waveform = load_audio_16k(wav_path)

    inputs = feature_extractor(waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device)).logits  # (1, T, C)
    log_probs = torch.log_softmax(logits, dim=-1)

    id_to_phoneme = {v: k for k, v in vocab.items()}
    blank_id = vocab["<pad>"]  # el token <pad> es el blank de CTC en este modelo

    # Reconocimiento LIBRE (greedy CTC decode, sin forzar ninguna palabra) --
    # sirve como referencia de qué escuchó el modelo posta, útil para debug.
    pred_ids = torch.argmax(log_probs, dim=-1)[0].tolist()
    reconocido_libre = _greedy_ctc_decode(pred_ids, id_to_phoneme, blank_id)

    fonemas_esperados = texto_a_fonemas(palabra)
    target_ids, fonemas_usados = _phoneme_to_token_ids(fonemas_esperados, vocab)
    if not target_ids:
        raise ValueError(f"No se pudo mapear ningún fonema de '{palabra}' al vocabulario del modelo.")

    targets = torch.tensor([target_ids], dtype=torch.int32, device=device)

    aligned_labels, aligned_scores = F.forced_align(
        log_probs, targets, blank=blank_id,
    )
    aligned_labels = aligned_labels[0].tolist()
    aligned_scores = aligned_scores[0].tolist()

    # Agrupa los frames alineados por fonema (cada fonema ocupa varios
    # frames consecutivos) y promedia el score dentro de cada grupo.
    scores_por_fonema = []
    idx_target = 0
    frame_scores_actual = []
    for label, score in zip(aligned_labels, aligned_scores):
        if label == blank_id:
            continue
        if idx_target < len(target_ids) and label == target_ids[idx_target]:
            frame_scores_actual.append(score)
        elif frame_scores_actual:
            # Cambió de fonema -- cierra el grupo anterior y arranca el siguiente.
            scores_por_fonema.append(float(np.mean(frame_scores_actual)))
            frame_scores_actual = [score]
            idx_target += 1
        else:
            frame_scores_actual = [score]
    if frame_scores_actual:
        scores_por_fonema.append(float(np.mean(frame_scores_actual)))

    # Completa con None si por algún motivo quedaron menos grupos que
    # fonemas esperados (clip cortado a mitad de palabra, etc.) -- mejor
    # marcarlo explícito que fingir un score que no se calculó de verdad.
    while len(scores_por_fonema) < len(fonemas_usados):
        scores_por_fonema.append(None)

    sospechosos = [
        i for i, s in enumerate(scores_por_fonema)
        if s is not None and s < GOP_SUSPICIOUS_THRESHOLD
    ]

    distancia = _distancia_edicion(fonemas_usados, reconocido_libre)
    normalizada = distancia / max(len(fonemas_usados), len(reconocido_libre), 1)
    palabra_reconocida = normalizada <= PALABRA_MISMATCH_THRESHOLD

    return {
        "fonemas": fonemas_usados,
        "scores": scores_por_fonema,
        "sospechosos": sospechosos,
        "reconocido_libre": " ".join(reconocido_libre),
        "sonido_reconocido": fonemas_a_texto_aproximado(reconocido_libre),
        "distancia_normalizada": normalizada,
        "palabra_reconocida": palabra_reconocida,
        "tipo_error_audio": diagnose_tipo_error(fonemas_usados, reconocido_libre),
    }


# Umbral de distancia de edición normalizada para decidir si la palabra
# objetivo realmente aparece en la frase reconocida libremente -- más
# permisivo que PALABRA_MISMATCH_THRESHOLD porque acá se compara contra la
# MEJOR ventana encontrada (ya recortada al largo de la palabra), no contra
# la frase entera.
FRASE_PALABRA_MISMATCH_THRESHOLD = 0.5

# Pedido explícito: si el chico dice solo la palabra suelta (o una frase
# mucho más corta que la pedida) en vez de la frase completa, NO tiene que
# contar como intento válido -- antes solo se validaba la palabra objetivo
# dentro de lo que sea que se haya grabado, así que decir nada más que
# "casa" ya pasaba como si hubiera dicho "mi casa es grande". Se compara el
# LARGO de lo reconocido libremente contra el largo esperado de fonemas de
# la frase entera (no la distancia de edición: el reconocimiento libre ya
# tiene ruido normal incluso en una frase bien dicha, pero el LARGO es un
# indicador mucho más robusto de si faltó decir una parte entera).
FRASE_COMPLETA_LARGO_MINIMO = 0.6


def _localizar_subsecuencia(secuencia, objetivo):
    """Busca dónde aparece `objetivo` (lista de fonemas de la palabra sola)
    dentro de `secuencia` (lista de fonemas de la frase completa), probando
    todas las ventanas de largo parecido y quedándose con la de menor
    distancia de edición. No exige coincidencia exacta a propósito: la
    regla de "r fuerte al principio de palabra" en texto_a_fonemas() mira
    el inicio del STRING completo, así que una palabra que empieza con r
    en medio de una frase puede fonemizarse levemente distinto que la
    misma palabra calculada sola -- por eso se busca la MEJOR ventana, no
    un substring exacto.

    Devuelve (inicio, fin) del mejor tramo dentro de `secuencia`."""
    n, m = len(secuencia), len(objetivo)
    if m == 0 or n == 0:
        return 0, 0
    mejor_inicio, mejor_fin, mejor_dist = 0, min(m, n), None
    for largo in (max(1, m - 1), m, m + 1):
        for inicio in range(0, max(1, n - largo + 1)):
            fin = inicio + largo
            if fin > n:
                continue
            dist = _distancia_edicion(secuencia[inicio:fin], objetivo)
            if mejor_dist is None or dist < mejor_dist:
                mejor_inicio, mejor_fin, mejor_dist = inicio, fin, dist
    return mejor_inicio, mejor_fin


def score_word_in_sentence(wav_path, frase, palabra_objetivo, model, feature_extractor, vocab, device):
    """Como score_pronunciation(), pero el chico dice una FRASE completa
    (ej. "el perro corre por las montañas") en vez de la palabra sola --
    pedido explícito: que se pueda practicar en un contexto más natural,
    no solo palabras sueltas. No usa el clasificador de audio entrenado
    (ver train_audio_classifier.py) porque ESE se entrenó únicamente con
    clips de la palabra sola: si se le da una frase entera, el embedding
    resume toda la frase junta y la clasificación no tiene sentido. Esto
    en cambio solo necesita el GOP genérico (wav2vec2 sin entrenamiento
    propio), que no depende de ningún dataset -- se alinea la frase
    ENTERA contra el audio, y después se recorta el resultado solo al
    tramo donde aparece `palabra_objetivo` dentro de esa frase.

    Devuelve el mismo formato que score_pronunciation(), más "frase" y
    "palabra_objetivo" para que el cliente sepa contra qué se comparó."""
    waveform = load_audio_16k(wav_path)

    inputs = feature_extractor(waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device)).logits
    log_probs = torch.log_softmax(logits, dim=-1)

    id_to_phoneme = {v: k for k, v in vocab.items()}
    blank_id = vocab["<pad>"]

    pred_ids = torch.argmax(log_probs, dim=-1)[0].tolist()
    reconocido_libre = _greedy_ctc_decode(pred_ids, id_to_phoneme, blank_id)

    fonemas_frase = texto_a_fonemas(frase)
    fonemas_palabra = texto_a_fonemas(palabra_objetivo)
    target_ids, fonemas_frase_usados = _phoneme_to_token_ids(fonemas_frase, vocab)
    if not target_ids:
        raise ValueError(f"No se pudo mapear ningún fonema de la frase '{frase}' al vocabulario del modelo.")

    targets = torch.tensor([target_ids], dtype=torch.int32, device=device)
    aligned_labels, aligned_scores = F.forced_align(log_probs, targets, blank=blank_id)
    aligned_labels = aligned_labels[0].tolist()
    aligned_scores = aligned_scores[0].tolist()

    # Además del score por fonema, se guarda en qué FRAME de audio empieza y
    # termina cada grupo -- hace falta para recortar la forma de onda más
    # abajo, no solo los fonemas/scores.
    scores_por_fonema = []
    frame_ranges = []
    idx_target = 0
    frame_scores_actual = []
    frame_indices_actual = []
    for frame_idx, (label, score) in enumerate(zip(aligned_labels, aligned_scores)):
        if label == blank_id:
            continue
        if idx_target < len(target_ids) and label == target_ids[idx_target]:
            frame_scores_actual.append(score)
            frame_indices_actual.append(frame_idx)
        elif frame_scores_actual:
            scores_por_fonema.append(float(np.mean(frame_scores_actual)))
            frame_ranges.append((frame_indices_actual[0], frame_indices_actual[-1]))
            frame_scores_actual = [score]
            frame_indices_actual = [frame_idx]
            idx_target += 1
        else:
            frame_scores_actual = [score]
            frame_indices_actual = [frame_idx]
    if frame_scores_actual:
        scores_por_fonema.append(float(np.mean(frame_scores_actual)))
        frame_ranges.append((frame_indices_actual[0], frame_indices_actual[-1]))
    while len(scores_por_fonema) < len(fonemas_frase_usados):
        scores_por_fonema.append(None)
        frame_ranges.append(None)

    # Recorta todo (fonemas y scores) al tramo de la frase donde está la
    # palabra objetivo -- de acá para abajo es igual que score_pronunciation
    # pero mirando solo ese pedacito, no la frase entera.
    inicio, fin = _localizar_subsecuencia(fonemas_frase_usados, fonemas_palabra)
    fonemas_usados = fonemas_frase_usados[inicio:fin]
    scores_recortados = scores_por_fonema[inicio:fin]

    # Recorta la forma de onda ORIGINAL al mismo tramo de tiempo, con un
    # colchón de 150ms de cada lado (el clasificador entrenado, ver
    # train_audio_classifier.py, aprendió de clips grabados como palabra
    # aislada, con algo de silencio natural alrededor -- recortar justo al
    # fonema sin margen suena distinto a esos clips de entrenamiento).
    # Frame -> muestra de audio: la relación es lineal (frames totales del
    # modelo vs. muestras totales del audio), no depende de una constante
    # fija por si el modelo cambia de arquitectura.
    rangos_validos = [r for r in frame_ranges[inicio:fin] if r is not None]
    audio_recortado = None
    if rangos_validos:
        total_frames = log_probs.shape[1]
        total_muestras = waveform.shape[0]
        muestras_por_frame = total_muestras / max(total_frames, 1)
        frame_inicio = min(r[0] for r in rangos_validos)
        frame_fin = max(r[1] for r in rangos_validos)
        colchon_muestras = int(0.15 * TARGET_SAMPLE_RATE)
        muestra_inicio = max(0, int(frame_inicio * muestras_por_frame) - colchon_muestras)
        muestra_fin = min(total_muestras, int((frame_fin + 1) * muestras_por_frame) + colchon_muestras)
        if muestra_fin > muestra_inicio:
            audio_recortado = waveform[muestra_inicio:muestra_fin]

    sospechosos = [
        i for i, s in enumerate(scores_recortados)
        if s is not None and s < GOP_SUSPICIOUS_THRESHOLD
    ]

    # Pedido explícito: hasta acá solo se revisaba la palabra objetivo
    # ("torre") -- si el resto de la frase pedida cambiaba (ej. decir "la
    # torre es muy BAJA" en vez de "ALTA"), igual contaba bien porque
    # "torre" se había dicho correctamente. Ahora se revisan también los
    # fonemas de la frase que quedan FUERA del tramo de la palabra objetivo
    # (ya alineados arriba contra la frase entera) -- si hay dos o más
    # fonemas sospechosos ahí afuera, lo más probable es que cambió una
    # palabra entera (una sola nota baja aislada puede ser ruido normal del
    # reconocimiento, dos o más seguidas en el mismo lugar no).
    scores_fuera_de_la_palabra = scores_por_fonema[:inicio] + scores_por_fonema[fin:]
    sospechosos_fuera = [
        s for s in scores_fuera_de_la_palabra
        if s is not None and s < GOP_SUSPICIOUS_THRESHOLD
    ]
    resto_de_la_frase_ok = len(sospechosos_fuera) < 2

    # Para "¿se reconoció la palabra?" se busca la mejor ventana DENTRO del
    # reconocimiento libre de la frase entera (mismo criterio de
    # _localizar_subsecuencia), no se compara la palabra contra la frase
    # completa reconocida (eso infla la distancia sin sentido).
    lib_inicio, lib_fin = _localizar_subsecuencia(reconocido_libre, fonemas_palabra)
    ventana_reconocida = reconocido_libre[lib_inicio:lib_fin]
    distancia = _distancia_edicion(ventana_reconocida, fonemas_palabra)
    normalizada = distancia / max(len(fonemas_palabra), len(ventana_reconocida), 1)
    palabra_reconocida = normalizada <= FRASE_PALABRA_MISMATCH_THRESHOLD

    # ¿Dijo la frase ENTERA, o solo un pedacito (ej. nada más que la
    # palabra suelta)? Ver comentario de FRASE_COMPLETA_LARGO_MINIMO arriba.
    frase_completa_dicha = (
        len(reconocido_libre) >= FRASE_COMPLETA_LARGO_MINIMO * len(fonemas_frase_usados)
    )

    # Guarda el audio recortado en un .wav temporal -- así el llamador
    # (server/main.py) puede pasárselo directo al clasificador ENTRENADO
    # (extract_embedding espera una ruta de archivo, no un tensor), y
    # combinar esa señal (99% de accuracy real) con el GOP de acá arriba,
    # en vez de depender solo del GOP para frases completas.
    audio_recortado_path = None
    if audio_recortado is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_recortado_path = tmp.name
        torchaudio.save(audio_recortado_path, audio_recortado.unsqueeze(0), TARGET_SAMPLE_RATE)

    return {
        "frase": frase,
        "palabra_objetivo": palabra_objetivo,
        "fonemas": fonemas_usados,
        "scores": scores_recortados,
        "sospechosos": sospechosos,
        "reconocido_libre": " ".join(reconocido_libre),
        "sonido_reconocido": fonemas_a_texto_aproximado(ventana_reconocida),
        "distancia_normalizada": normalizada,
        "palabra_reconocida": palabra_reconocida,
        "frase_completa_dicha": frase_completa_dicha,
        "resto_de_la_frase_ok": resto_de_la_frase_ok,
        "tipo_error_audio": diagnose_tipo_error(fonemas_palabra, reconocido_libre),
        "audio_recortado_path": audio_recortado_path,
    }


def print_report(palabra, resultado):
    print(f"\nPalabra objetivo: '{palabra}'")
    print(f"Fonemas esperados: {' '.join(resultado['fonemas'])}")
    print(f"Lo que reconoció el modelo (libre, sin forzar): {resultado['reconocido_libre']}")
    print("Score por fonema (más alto = más claro, más bajo = más dudoso):")
    for i, (fon, score) in enumerate(zip(resultado["fonemas"], resultado["scores"])):
        marca = " <-- sospechoso" if i in resultado["sospechosos"] else ""
        score_str = f"{score:.2f}" if score is not None else "N/A"
        print(f"  {fon:>4s}: {score_str}{marca}")
    if resultado["sospechosos"]:
        letras = [resultado["fonemas"][i] for i in resultado["sospechosos"]]
        print(f"\n-> Posible error de pronunciación en: {', '.join(letras)}")
    else:
        print("\n-> Todos los fonemas dentro de lo esperado.")
    if not resultado["palabra_reconocida"]:
        print(f"-> OJO: lo reconocido libremente se parece poco a '{palabra}' "
              f"(distancia normalizada {resultado['distancia_normalizada']:.2f}) -- "
              "probablemente dijo otra palabra.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python audio_pronunciation.py <ruta_al_wav> <palabra>")
        sys.exit(1)

    wav_path, palabra = sys.argv[1], sys.argv[2]
    model, feature_extractor, vocab, device = load_model()
    resultado = score_pronunciation(wav_path, palabra, model, feature_extractor, vocab, device)
    print_report(palabra, resultado)
