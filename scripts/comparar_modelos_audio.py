"""Compara modelos de audio y reglas de decision con las etiquetas que puso una PERSONA (pantalla #/etiquetar).

Para cada modelo mide, con validacion cruzada (una palabra nunca esta a la vez en entrenar y en probar):
  - la regla actual del sistema (la que quedo escrita en el nombre del archivo),
  - una regla aprendida con los puntajes de cada sonido (GOP),
  - un clasificador por sonido: puntajes + la "huella" del sonido mas flojo de la palabra.
Muestra exactitud, falsos aprobados (era mal y dijo bien) y falsos rechazos. Lo importante en terapia son los falsos aprobados.

Uso:
  .venv-fonokids\\Scripts\\python.exe scripts\\comparar_modelos_audio.py
  .venv-fonokids\\Scripts\\python.exe scripts\\comparar_modelos_audio.py --modelos actual es mms
  .venv-fonokids\\Scripts\\python.exe scripts\\comparar_modelos_audio.py --guardar        (guarda el mejor en models/clasificador_audio.joblib)

Modelos: "actual" (xlsr-53 espeak, el que usa la app), "es" (wav2vec2 xlsr-53 en español, ~1.2 GB) y "mms" (MMS-1b-all de Meta,
~4 GB; se baja la primera vez). No cambia nada del servidor: primero se revisa el resultado en data/comparacion_modelos_audio.json.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))

MODELOS = {
    "actual": "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "mms": "facebook/mms-1b-all",
}


def leer_etiquetas(ruta, carpeta):
    filas = {}
    with open(ruta, encoding="utf-8") as fh:
        for linea in fh.read().splitlines()[1:]:
            p = linea.split(";")
            if len(p) >= 3 and p[2] in ("bien", "mal"):
                filas[p[0]] = p[2]  # vale la ultima etiqueta de cada archivo
    out = []
    for archivo, et in filas.items():
        m = re.match(r"\d{8}_\d{6}_(.+?)_ok\d+_(bien|mal)\.wav", archivo)
        ruta_wav = os.path.join(carpeta, archivo)
        if m and os.path.exists(ruta_wav):
            out.append({"archivo": archivo, "ruta": ruta_wav, "palabra": m.group(1), "humano": 1 if et == "bien" else 0,
                        "sistema": 1 if m.group(2) == "bien" else 0})
    return out


def cargar(clave):
    """Devuelve (modelo, extractor, vocab, dispositivo, es_fonetico)."""
    import torch
    if clave == "actual":
        from audio_pronunciation import load_model
        m, fe, vocab, dev = load_model()
        return m, fe, vocab, dev, True
    from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2FeatureExtractor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nombre = MODELOS[clave]
    fe = Wav2Vec2FeatureExtractor.from_pretrained(nombre)
    m = AutoModelForCTC.from_pretrained(nombre)
    if clave == "mms":  # MMS trae un adaptador por idioma: se elige el español
        proc = AutoProcessor.from_pretrained(nombre)
        proc.tokenizer.set_target_lang("spa")
        m.load_adapter("spa")
        vocab = proc.tokenizer.get_vocab()
    else:
        from transformers import Wav2Vec2CTCTokenizer
        vocab = Wav2Vec2CTCTokenizer.from_pretrained(nombre).get_vocab()
    return m.to(dev).eval(), fe, vocab, dev, False


def tokens_de(palabra, vocab, fonetico):
    if fonetico:
        from audio_pronunciation import texto_a_fonemas, _phoneme_to_token_ids
        ids, _ = _phoneme_to_token_ids(texto_a_fonemas(palabra), vocab)
        return ids
    letras = [c for c in palabra.lower() if not c.isspace()]
    mayus = "a" not in vocab and "A" in vocab
    ids = []
    for c in letras:
        c = c.upper() if mayus else c
        if c in vocab:
            ids.append(vocab[c])
    return ids


def extraer(clave, muestras, cache):
    """Por grabacion: puntaje de cada sonido (GOP) y la huella del modelo en el sonido mas flojo y en toda la palabra."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        previo = {a: (g, w, f) for a, g, w, f in zip(z["archivos"], z["gops"], z["huella_peor"], z["huella_total"])}
    else:
        previo = {}
    import torch
    import torchaudio
    from audio_pronunciation import load_audio_16k, recortar_ruido_inicial
    modelo = fe = vocab = dev = fon = None
    for i, s in enumerate(muestras):
        if s["archivo"] in previo:
            continue
        if modelo is None:
            modelo, fe, vocab, dev, fon = cargar(clave)
            blanco = vocab.get("<pad>", 0)
        wav = recortar_ruido_inicial(load_audio_16k(s["ruta"]))
        ent = fe(wav.numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(dev)
        try:
            with torch.inference_mode():
                sal = modelo(ent, output_hidden_states=True)
            lp = torch.log_softmax(sal.logits, dim=-1)
            ids = tokens_de(s["palabra"], vocab, fon)
            if not ids:
                raise ValueError("sin sonidos")
            etiq, pts = torchaudio.functional.forced_align(lp, torch.tensor([ids], dtype=torch.int32, device=dev), blank=blanco)
            tramos = torchaudio.functional.merge_tokens(etiq[0], pts[0])
            oculto = sal.hidden_states[-8][0].cpu().numpy()  # capa intermedia: guarda mas informacion de sonidos
            lp1 = lp[0].cpu().numpy()
            gops, spans = [], []
            for t in tramos:
                gops.append(float(np.mean([lp1[f, t.token] for f in range(t.start, t.end)])))
                spans.append((t.start, t.end))
            while len(gops) < len(ids):
                gops.append(-12.0)
                spans.append((0, oculto.shape[0]))
            peor = int(np.argmin(gops))
            a, b = spans[peor]
            previo[s["archivo"]] = (np.array(gops), oculto[a:b].mean(0), oculto.mean(0))
        except Exception as e:  # noqa: BLE001
            print("  se salta", s["archivo"], e)
        if (i + 1) % 20 == 0:
            print(f"  {clave}: {i + 1}/{len(muestras)}")
    if previo:
        a = list(previo)
        np.savez(cache, archivos=np.array(a), gops=np.array([previo[k][0] for k in a], dtype=object),
                 huella_peor=np.array([previo[k][1] for k in a]), huella_total=np.array([previo[k][2] for k in a]))
    return previo


def resumen_gops(g):
    s = np.maximum(np.array(g, dtype=float), -12.0)
    return [s.min(), s.mean(), (s < -4).mean(), (s < -2).mean(), s[0], s[-1]]


def informe(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    return {"exactitud": round(float((pred == y).mean() * 100), 1),
            "falsos_aprobados": int(((pred == 1) & (y == 0)).sum()),
            "falsos_rechazos": int(((pred == 0) & (y == 1)).sum())}


def validar(X, y, grupos, con_pca=0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pasos = [StandardScaler()]
    if con_pca:
        pasos.append(PCA(n_components=min(con_pca, len(y) // 3, X.shape[1])))
    pasos.append(LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
    clf = make_pipeline(*pasos)
    try:
        cv = GroupKFold(n_splits=min(5, len(set(grupos))))
        pred = cross_val_predict(clf, X, y, cv=cv, groups=grupos)
    except Exception:  # noqa: BLE001
        pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return pred, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiquetas", default=os.path.join(RAIZ, "data", "etiquetas_audio.csv"))
    ap.add_argument("--carpeta", default=os.path.join(RAIZ, "data", "audios_para_etiquetar"))
    ap.add_argument("--modelos", nargs="+", default=["actual"], choices=list(MODELOS))
    ap.add_argument("--minimo", type=int, default=40)
    ap.add_argument("--guardar", action="store_true", help="guarda el mejor clasificador en models/clasificador_audio.joblib")
    a = ap.parse_args()

    if not os.path.exists(a.etiquetas):
        sys.exit("Todavia no hay etiquetas: abre http://localhost:5173/#/etiquetar y marca las grabaciones.")
    muestras = leer_etiquetas(a.etiquetas, a.carpeta)
    print(f"{len(muestras)} grabaciones etiquetadas como bien o mal")
    if len(muestras) < a.minimo:
        sys.exit(f"Faltan etiquetas: hay {len(muestras)} y se necesitan al menos {a.minimo}. Sigue etiquetando.")
    y = np.array([m["humano"] for m in muestras])
    if min(y.sum(), (1 - y).sum()) < 8:
        sys.exit("Hace falta que haya al menos 8 grabaciones bien y 8 mal para poder comparar.")
    grupos = [m["palabra"] for m in muestras]
    actual = informe([m["sistema"] for m in muestras], y)
    print("Regla actual del sistema:", actual)

    os.makedirs(os.path.join(RAIZ, "data"), exist_ok=True)
    resultados, mejor = {"regla_actual": actual, "grabaciones": len(y), "modelos": {}}, None
    for clave in a.modelos:
        print(f"\n== Modelo {clave}: {MODELOS[clave]}")
        try:
            f = extraer(clave, muestras, os.path.join(RAIZ, "data", f"_cache_audio_{clave}.npz"))
        except Exception as e:  # noqa: BLE001
            print("  no se pudo usar este modelo:", e)
            continue
        ok = [i for i, m in enumerate(muestras) if m["archivo"] in f]
        yy, gg = y[ok], [grupos[i] for i in ok]
        Xg = np.array([resumen_gops(f[muestras[i]["archivo"]][0]) for i in ok])
        Xp = np.array([f[muestras[i]["archivo"]][1] for i in ok])
        Xt = np.array([f[muestras[i]["archivo"]][2] for i in ok])
        r = {}
        for nombre, X, pca in (("regla_aprendida_gop", Xg, 0),
                               ("clasificador_por_sonido", np.hstack([Xg, Xp]), 12),
                               ("clasificador_palabra_completa", np.hstack([Xg, Xt]), 12)):
            pred, clf = validar(X, yy, gg, pca)
            r[nombre] = informe(pred, yy)
            print(f"  {nombre}: {r[nombre]}")
            # Criterio de eleccion: menos falsos aprobados primero y despues exactitud.
            clave_ord = (r[nombre]["exactitud"] - 2 * r[nombre]["falsos_aprobados"])
            if mejor is None or clave_ord > mejor[0]:
                mejor = (clave_ord, clave, nombre, X, yy, clf)
        resultados["modelos"][clave] = r

    if mejor:
        _, clave, nombre, X, yy, clf = mejor
        resultados["mejor"] = {"modelo": clave, "metodo": nombre, **resultados["modelos"][clave][nombre]}
        dif = resultados["mejor"]["exactitud"] - actual["exactitud"]
        print(f"\nMejor combinacion: {clave} + {nombre} ({resultados['mejor']['exactitud']}%). Diferencia con la regla actual: {dif:+.1f} puntos.")
        print("Vale la pena adoptarla." if dif >= 3 and resultados["mejor"]["falsos_aprobados"] <= actual["falsos_aprobados"]
              else "No hay mejora clara que justifique cambiar: sigue la regla actual y junta mas etiquetas (de varios ninos).")
        if a.guardar:
            import joblib
            clf.fit(X, yy)
            os.makedirs(os.path.join(RAIZ, "models"), exist_ok=True)
            joblib.dump({"modelo": clave, "metodo": nombre, "clasificador": clf}, os.path.join(RAIZ, "models", "clasificador_audio.joblib"))
            print("Guardado en models/clasificador_audio.joblib (el servidor todavia no lo usa).")
    with open(os.path.join(RAIZ, "data", "comparacion_modelos_audio.json"), "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)
    print("Resultados en data/comparacion_modelos_audio.json")


if __name__ == "__main__":
    main()
