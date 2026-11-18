"""Afina el criterio del modelo de audio con las etiquetas que puso una PERSONA (pantalla #/etiquetar).

Para cada grabacion etiquetada saca lo que mide el modelo (puntaje de cada sonido de la palabra) y prueba si una regla
aprendida de esas etiquetas acierta mas que la regla actual. Con validacion cruzada, para no engañarse.

Uso:   .venv-fonokids\\Scripts\\python.exe scripts\\afinar_audio.py
       .venv-fonokids\\Scripts\\python.exe scripts\\afinar_audio.py --etiquetas otro.csv --minimo 40

Escribe data/afinado_audio.json con los resultados (exactitud, falsos aprobados, falsos rechazos y los pesos aprendidos).
No cambia nada del servidor por si solo: primero se revisa el resultado.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))


def caracteristicas(scores):
    s = np.array([-12.0 if x is None else max(float(x), -12.0) for x in scores])
    return [float(s.min()), float(s.mean()), float((s < -4.0).mean()), float((s < -2.0).mean()), float(s[0]), float(s[-1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiquetas", default=os.path.join(RAIZ, "data", "etiquetas_audio.csv"))
    ap.add_argument("--carpeta", default=os.path.join(RAIZ, "data", "audios_para_etiquetar"))
    ap.add_argument("--minimo", type=int, default=40, help="minimo de grabaciones etiquetadas (bien o mal) para afinar")
    a = ap.parse_args()

    if not os.path.exists(a.etiquetas):
        sys.exit("Todavia no hay etiquetas: abre http://localhost:5173/#/etiquetar y marca las grabaciones.")
    filas = []
    with open(a.etiquetas, encoding="utf-8") as fh:
        for linea in fh.read().splitlines()[1:]:
            p = linea.split(";")
            if len(p) >= 3 and p[2] in ("bien", "mal"):
                filas.append((p[0], p[2]))
    filas = list({f: e for f, e in filas}.items())  # ultima etiqueta de cada archivo
    print(f"{len(filas)} grabaciones etiquetadas como bien o mal")
    if len(filas) < a.minimo:
        sys.exit(f"Faltan etiquetas: hay {len(filas)} y se necesitan al menos {a.minimo}. Sigue etiquetando.")

    import torch
    from audio_pronunciation import load_model, score_pronunciation
    modelo, fe, vocab = load_model()[:3]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, sistema, grupos = [], [], [], []
    for archivo, et in filas:
        ruta = os.path.join(a.carpeta, archivo)
        m = re.match(r"\d{8}_\d{6}_(.+?)_ok\d+_(bien|mal)\.wav", archivo)
        if not os.path.exists(ruta) or not m:
            continue
        try:
            r = score_pronunciation(ruta, m.group(1), modelo, fe, vocab, dev)
        except Exception as e:  # noqa: BLE001
            print("  se salta", archivo, e)
            continue
        X.append(caracteristicas(r["scores"]))
        y.append(1 if et == "bien" else 0)
        sistema.append(1 if m.group(2) == "bien" else 0)
        grupos.append(m.group(1))
    X, y, sistema = np.array(X), np.array(y), np.array(sistema)
    print(f"{len(y)} grabaciones usadas ({int(y.sum())} bien, {int((1 - y).sum())} mal)")

    def informe(pred):
        return {
            "exactitud": round(float((pred == y).mean() * 100), 1),
            "falsos_aprobados": int(((pred == 1) & (y == 0)).sum()),
            "falsos_rechazos": int(((pred == 0) & (y == 1)).sum()),
        }

    actual = informe(sistema)
    print("Regla actual del sistema:", actual)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
    try:
        cv = GroupKFold(n_splits=min(5, len(set(grupos))))  # que una palabra no este a la vez en entrenar y en probar
        pred = cross_val_predict(clf, X, y, cv=cv, groups=grupos)
    except Exception:  # noqa: BLE001
        pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    aprendida = informe(pred)
    print("Regla aprendida de tus etiquetas (validacion cruzada):", aprendida)

    clf.fit(X, y)
    salida = {
        "grabaciones": int(len(y)), "regla_actual": actual, "regla_aprendida": aprendida,
        "caracteristicas": ["min", "media", "frac<-4", "frac<-2", "primer_sonido", "ultimo_sonido"],
        "pesos": clf[-1].coef_[0].tolist(), "sesgo": float(clf[-1].intercept_[0]),
        "media": clf[0].mean_.tolist(), "desvio": clf[0].scale_.tolist(),
    }
    with open(os.path.join(RAIZ, "data", "afinado_audio.json"), "w", encoding="utf-8") as fh:
        json.dump(salida, fh, ensure_ascii=False, indent=2)
    mejora = aprendida["exactitud"] - actual["exactitud"]
    print(f"\nDiferencia: {mejora:+.1f} puntos. " + ("La regla aprendida mejora: vale la pena adoptarla." if mejora >= 3 else "No hay mejora clara: deja la regla actual y junta mas etiquetas."))
    print("Resultados en data/afinado_audio.json")


if __name__ == "__main__":
    main()
