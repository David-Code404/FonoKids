import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
TARGET_SAMPLE_RATE = 16000

GOP_SUSPICIOUS_THRESHOLD = -3.0

PALABRA_MISMATCH_THRESHOLD = 0.5


VOCALES_ACENTUADAS = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u"}


def texto_a_fonemas(palabra):
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
            fonemas.append("s")
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
            i += 2
            continue

        if c == "g" and nxt == "u" and i + 2 < n and palabra[i + 2] in ("e", "i"):
            fonemas.append("ɡ")
            i += 2
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
            fonemas.append("ʝ")
            i += 2
            continue

        if c == "y":
            es_vocal_final = (i == n - 1) and prev in "aeiou"
            fonemas.append("i" if es_vocal_final else "ʝ")
            i += 1
            continue

        if c == "z":
            fonemas.append("s")
            i += 1
            continue

        if c == "j":
            fonemas.append("x")
            i += 1
            continue

        if c == "h":
            i += 1
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

        i += 1

    return fonemas


def load_model(device=None):
    import json
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando {MODEL_NAME} (primera vez tarda, baja ~1.2GB)...")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

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
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SAMPLE_RATE)
    return waveform.squeeze(0)


EMBEDDING_DIM = 1024


def extract_embedding(wav_path, model, feature_extractor, device):
    waveform = load_audio_16k(wav_path)
    inputs = feature_extractor(waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        hidden = model.wav2vec2(inputs.input_values.to(device)).last_hidden_state
        pooled = hidden.mean(dim=1).squeeze(0)
    return pooled.cpu().numpy()


def _phoneme_to_token_ids(fonemas, vocab):
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
    fonemas = []
    anterior = None
    for tid in pred_ids:
        if tid != anterior and tid != blank_id:
            fonemas.append(id_to_phoneme.get(tid, "?"))
        anterior = tid
    return fonemas


_FONEMA_A_LETRA = {"r": "rr", "ɾ": "r", "x": "j", "ɡ": "g", "tʃ": "ch", "ʝ": "ll", "ɲ": "ñ"}


def fonemas_a_texto_aproximado(fonemas):
    return "".join(_FONEMA_A_LETRA.get(f, f) for f in fonemas)


def _distancia_edicion(a, b):
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


def _align_ops(a, b):
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


_SUSTITUCIONES_CONOCIDAS = {
    "r": {"l": "lambdacismo", "d": "dentalizacion"},
    "ɾ": {"l": "lambdacismo", "d": "dentalizacion"},
    "k": {"t": "anteriorizacion", "x": "debilitamiento", "h": "debilitamiento"},
    "ɡ": {"d": "anteriorizacion", "x": "debilitamiento", "h": "debilitamiento"},
    "s": {"θ": "interdental", "t": "oclusion", "d": "oclusion"},
    "ʝ": {"l": "distorsion", "i": "debilitamiento"},
    "tʃ": {"ʃ": "distorsion", "t": "debilitamiento", "s": "debilitamiento"},
}
_SINFONES = {"bl", "kl", "fl", "ɡl", "pl"}


def diagnose_tipo_error(fonemas_esperados, reconocido_libre_lista):
    ops = _align_ops(fonemas_esperados, reconocido_libre_lista)

    for idx, (tipo, esperado, reconocido) in enumerate(ops):
        if esperado == "l" and idx > 0:
            anterior = ops[idx - 1][1]
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
    waveform = load_audio_16k(wav_path)

    inputs = feature_extractor(waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device)).logits
    log_probs = torch.log_softmax(logits, dim=-1)

    id_to_phoneme = {v: k for k, v in vocab.items()}
    blank_id = vocab["<pad>"]

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

    scores_por_fonema = []
    idx_target = 0
    frame_scores_actual = []
    for label, score in zip(aligned_labels, aligned_scores):
        if label == blank_id:
            continue
        if idx_target < len(target_ids) and label == target_ids[idx_target]:
            frame_scores_actual.append(score)
        elif frame_scores_actual:
            scores_por_fonema.append(float(np.mean(frame_scores_actual)))
            frame_scores_actual = [score]
            idx_target += 1
        else:
            frame_scores_actual = [score]
    if frame_scores_actual:
        scores_por_fonema.append(float(np.mean(frame_scores_actual)))

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


FRASE_PALABRA_MISMATCH_THRESHOLD = 0.5

FRASE_COMPLETA_LARGO_MINIMO = 0.6


def _localizar_subsecuencia(secuencia, objetivo):
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

    inicio, fin = _localizar_subsecuencia(fonemas_frase_usados, fonemas_palabra)
    fonemas_usados = fonemas_frase_usados[inicio:fin]
    scores_recortados = scores_por_fonema[inicio:fin]

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

    scores_fuera_de_la_palabra = scores_por_fonema[:inicio] + scores_por_fonema[fin:]
    sospechosos_fuera = [
        s for s in scores_fuera_de_la_palabra
        if s is not None and s < GOP_SUSPICIOUS_THRESHOLD
    ]
    resto_de_la_frase_ok = len(sospechosos_fuera) < 2

    lib_inicio, lib_fin = _localizar_subsecuencia(reconocido_libre, fonemas_palabra)
    ventana_reconocida = reconocido_libre[lib_inicio:lib_fin]
    distancia = _distancia_edicion(ventana_reconocida, fonemas_palabra)
    normalizada = distancia / max(len(fonemas_palabra), len(ventana_reconocida), 1)
    palabra_reconocida = normalizada <= FRASE_PALABRA_MISMATCH_THRESHOLD

    frase_completa_dicha = (
        len(reconocido_libre) >= FRASE_COMPLETA_LARGO_MINIMO * len(fonemas_frase_usados)
    )

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
