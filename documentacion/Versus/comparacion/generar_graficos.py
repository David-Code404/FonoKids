"""
generar_graficos.py
-------------------
Dibuja los graficos de comparacion usando SOLO numeros publicados en los
papers que estan en documentacion/investigacion/ y documentacion/Versus/.
No entrena nada ni toca los modelos del proyecto.

Cada lista de datos lleva comentada la tabla y la pagina del PDF de donde
sale, para poder verificarla.

Uso:  python generar_graficos.py   ->  graficos/*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "graficos")
os.makedirs(OUT, exist_ok=True)

TEAL = "#0f766e"      # lo que usa FonoKids (o su familia)
GRIS = "#b9b5cf"      # otros metodos
MORADO = "#6c5ce7"
NARANJA = "#e07a2f"
TXT = "#1f1d2b"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#8b87a3", "axes.labelcolor": TXT,
    "xtick.color": TXT, "ytick.color": TXT, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def guardar(fig, nombre, fuente):
    fig.text(0.01, -0.02, "Fuente: " + fuente, fontsize=8, color="#6b6880", ha="left", va="top")
    fig.savefig(os.path.join(OUT, nombre), dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("ok", nombre)


def barras_h(ax, nombres, valores, destacar, titulo, xlabel, fmt="{:.2f}", menor_mejor=True):
    y = np.arange(len(nombres))
    cols = [TEAL if n in destacar else GRIS for n in nombres]
    ax.barh(y, valores, color=cols, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(nombres)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel + ("  (menor es mejor)" if menor_mejor else "  (mayor es mejor)"))
    ax.set_title(titulo, loc="left", fontweight="bold", fontsize=12.5)
    for yi, v in zip(y, valores):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=10)
    ax.set_xlim(0, max(valores) * 1.15)


# ===================================================================== 1
# wav2vec 2.0, Table 2 (pag. 7): WER LibriSpeech 960h etiquetadas, con LM. test-clean / test-other
def g_wav2vec_librispeech():
    filas = [
        ("CTC Transformer", 2.47, 5.45),
        ("S2S Transformer", 2.33, 5.17),
        ("Transformer Transducer", 2.0, 4.6),
        ("ContextNet", 1.9, 4.1),
        ("Conformer", 1.9, 3.9),
        ("wav2vec 2.0 LARGE (desde cero)", 2.1, 4.6),
        ("wav2vec 2.0 BASE (LS-960)", 2.1, 4.8),
        ("wav2vec 2.0 LARGE (LS-960)", 2.0, 4.1),
        ("wav2vec 2.0 LARGE (LV-60k)", 1.8, 3.3),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, idx, tit in zip(axs, (1, 2), ("test-clean", "test-other")):
        nombres = [f[0] for f in filas]
        barras_h(ax, nombres, [f[idx] for f in filas], {n for n in nombres if "wav2vec" in n},
                 f"WER en {tit}", "WER (%)", "{:.2f}")
    axs[1].set_ylabel("")
    fig.suptitle("wav2vec 2.0 frente a otros modelos de reconocimiento de voz (LibriSpeech 960 h)",
                 fontweight="bold", fontsize=14, x=0.01, ha="left")
    guardar(fig, "01_wav2vec2_vs_otros_librispeech.png",
            "Baevski et al., wav2vec 2.0, Tabla 2 (pag. 7). Todos con modelo de lenguaje externo. En verde: la familia wav2vec 2.0 (tu app usa la variante multilingue XLSR-53 de esta familia).")


# wav2vec 2.0, Table 1 (pag. 6): test-other, con muy pocos datos etiquetados
def g_wav2vec_pocos_datos():
    x = ["10 min", "1 hora", "10 horas", "100 horas"]
    series = [
        ("Discrete BERT (4-gram)", [25.2, 17.6, 14.1, 12.1], GRIS, "o--"),
        ("wav2vec 2.0 BASE (LS-960, Transf.)", [12.9, 9.3, 7.8, 6.3], MORADO, "s-"),
        ("wav2vec 2.0 LARGE (LS-960, Transf.)", [10.8, 7.6, 6.1, 5.0], NARANJA, "s-"),
        ("wav2vec 2.0 LARGE (LV-60k, Transf.)", [8.2, 5.8, 4.9, 4.0], TEAL, "s-"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for nombre, v, c, st in series:
        ax.plot(x, v, st, color=c, label=nombre, lw=2.2, ms=7)
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:g}", (xi, vi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=c)
    ax.set_ylabel("WER en test-other (%)  (menor es mejor)")
    ax.set_xlabel("Cantidad de audio etiquetado usado para ajustar el modelo")
    ax.set_title("Con poquisimos datos etiquetados, wav2vec 2.0 sigue funcionando", loc="left", fontweight="bold", fontsize=13)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    guardar(fig, "02_wav2vec2_pocos_datos.png",
            "Baevski et al., wav2vec 2.0, Tabla 1 (pag. 6), columna test-other. Relevante para tu caso: dataset propio chico.")


# wav2vec 2.0, Table 3 (pag. 8): PER en TIMIT (test)
def g_timit():
    nombres = ["CNN + TD-filterbanks", "PASE+", "Li-GRU + fMLLR", "wav2vec", "vq-wav2vec", "wav2vec 2.0 LARGE"]
    vals = [18.0, 17.2, 14.9, 14.7, 11.6, 8.3]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    barras_h(ax, nombres, vals, {"wav2vec 2.0 LARGE"},
             "Error de fonemas (PER) en TIMIT, set de test", "PER (%)", "{:.1f}")
    guardar(fig, "03_wav2vec2_fonemas_timit.png",
            "Baevski et al., wav2vec 2.0, Tabla 3 (pag. 8). Es la tarea mas parecida a tu GOP: reconocer fonemas.")


# ===================================================================== 2
# Conformer, Table 2 (pag. 3): parametros vs WER test-other sin LM
def g_conformer_params():
    pts = [
        ("QuartzNet", 19, 11.28, "otro"), ("Transformer Transducer", 139, 5.6, "otro"),
        ("ContextNet S", 10.8, 7.0, "otro"), ("ContextNet M", 31.4, 5.4, "otro"), ("ContextNet L", 112.7, 4.6, "otro"),
        ("Conformer S", 10.3, 6.3, "conf"), ("Conformer M", 30.7, 5.0, "conf"), ("Conformer L", 118.8, 4.3, "conf"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for n, p, w, k in pts:
        ax.scatter(p, w, s=120, color=TEAL if k == "conf" else GRIS, zorder=3, edgecolor="white")
        ax.annotate(n, (p, w), textcoords="offset points", xytext=(8, 6), fontsize=9.5)
    conf = [(p, w) for n, p, w, k in pts if k == "conf"]
    ax.plot(*zip(*conf), color=TEAL, lw=1.6, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Parametros (millones, escala logaritmica)")
    ax.set_ylabel("WER en test-other, sin LM (%)  (menor es mejor)")
    ax.set_title("Conformer: mejor precision con menos parametros", loc="left", fontweight="bold", fontsize=13)
    ax.grid(alpha=0.25)
    guardar(fig, "04_conformer_precision_vs_tamano.png",
            "Gulati et al., Conformer, Tabla 2 (pag. 3). Tu LipReadingConformer usa este tipo de bloque (2 capas, 8 cabezas, kernel 15).")


# Conformer, Table 3 (pag. 4): ablacion, test-other sin LM
def g_conformer_ablacion():
    nombres = ["Conformer completo", "sin SWISH (usa ReLU)", "sin bloque convolucional", "sin FFN Macaron", "sin pos. relativa"]
    vals = [4.3, 4.5, 4.9, 5.0, 5.6]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    barras_h(ax, nombres, vals, {"Conformer completo"},
             "Que pasa si le quitas partes al Conformer (test-other)", "WER (%)", "{:.1f}")
    guardar(fig, "05_conformer_ablacion.png",
            "Gulati et al., Conformer, Tabla 3 (pag. 4). Quitar la convolucion empeora de 4.3 a 4.9: por eso el modulo convolucional.")


# ===================================================================== 3
# Bai et al., TCN vs recurrentes, Table 1 (pag. 6)
def g_tcn():
    modelos = ["LSTM", "GRU", "RNN", "TCN"]
    cols = ["#cfcce3", "#a19dc4", "#76709b", TEAL]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))

    tareas = ["MNIST secuencial", "MNIST permutado"]
    datos = {"LSTM": [87.2, 85.7], "GRU": [96.2, 87.3], "RNN": [21.5, 25.3], "TCN": [99.0, 97.2]}
    w = 0.2
    for i, m in enumerate(modelos):
        axs[0].bar(np.arange(2) + (i - 1.5) * w, datos[m], w, color=cols[i], label=m, edgecolor="white")
        for xi, v in zip(np.arange(2) + (i - 1.5) * w, datos[m]):
            axs[0].text(xi, v + 1, f"{v:g}", ha="center", fontsize=8)
    axs[0].set_xticks(range(2)); axs[0].set_xticklabels(tareas)
    axs[0].set_ylabel("Accuracy (%)  (mayor es mejor)"); axs[0].set_ylim(0, 125)
    axs[0].set_title("Clasificar secuencias", loc="left", fontweight="bold")
    axs[0].legend(frameon=False, ncol=4, fontsize=9, loc="upper center", handlelength=1.2)

    tareas = ["Char PTB", "text8"]
    datos = {"LSTM": [1.36, 1.50], "GRU": [1.37, 1.53], "RNN": [1.48, 1.69], "TCN": [1.31, 1.45]}
    for i, m in enumerate(modelos):
        axs[1].bar(np.arange(2) + (i - 1.5) * w, datos[m], w, color=cols[i])
        for xi, v in zip(np.arange(2) + (i - 1.5) * w, datos[m]):
            axs[1].text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axs[1].set_xticks(range(2)); axs[1].set_xticklabels(tareas)
    axs[1].set_ylabel("bits por caracter  (menor es mejor)"); axs[1].set_ylim(1.0, 1.85)
    axs[1].set_title("Modelado de texto", loc="left", fontweight="bold")

    tareas = ["Copy memory\n(T=1000)", "Adding problem\n(T=600)"]
    datos = {"LSTM": [0.0204, 0.164], "GRU": [0.0197, 5.3e-5], "RNN": [0.0202, 0.177], "TCN": [3.5e-5, 5.8e-5]}
    for i, m in enumerate(modelos):
        axs[2].bar(np.arange(2) + (i - 1.5) * w, datos[m], w, color=cols[i])
    axs[2].set_yscale("log")
    axs[2].set_xticks(range(2)); axs[2].set_xticklabels(tareas)
    axs[2].set_ylabel("loss, escala log  (menor es mejor)")
    axs[2].set_title("Memoria de largo plazo", loc="left", fontweight="bold")

    fig.suptitle("TCN frente a LSTM, GRU y RNN en tareas de secuencias", fontweight="bold", fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    guardar(fig, "06_tcn_vs_lstm_gru.png",
            "Bai, Kolter, Koltun, Tabla 1 (pag. 6). El TCN gana en las 3 familias de tareas. Tu video usa 3 bloques TCN antes del Conformer.")


# ===================================================================== 4
# HRNet, Table 23 (pag. 21): 300W full NME. Table 20 (pag. 20): WFLW test NME
def g_hrnet():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.0))
    n1 = ["RCN", "DSRN", "PCD-CNN", "CPM + SBR", "SAN", "DAN", "LAB (con info extra)", "DCFE (con 3D)", "HRNetV2-W18"]
    v1 = [5.41, 5.21, 4.44, 4.10, 3.98, 3.59, 3.49, 3.24, 3.32]
    barras_h(axs[0], n1, v1, {"HRNetV2-W18"}, "300W (68 puntos), conjunto completo", "NME (%)", "{:.2f}")
    n2 = ["ESR", "SDM", "CFSS", "DVLN", "LAB (con info extra)", "PDB (mas aumento)", "HRNetV2-W18"]
    v2 = [11.13, 10.29, 9.07, 6.08, 5.27, 5.11, 4.60]
    barras_h(axs[1], n2, v2, {"HRNetV2-W18"}, "WFLW (98 puntos), test", "NME (%)", "{:.2f}")
    fig.suptitle("HRNet frente a otros detectores de landmarks faciales", fontweight="bold", fontsize=14, x=0.01, ha="left")
    guardar(fig, "07_hrnet_landmarks_faciales.png",
            "Wang et al., HRNet, Tablas 23 (pag. 21) y 20 (pag. 20). NME = error normalizado. Tu pipeline saca los 20 puntos de labios de un detector de 68 puntos.")


# ===================================================================== 5
# LipFormer, Table V (GRID) y IV (CMLR), pag. 7
def g_lipformer():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.0))
    metodos = ["LipNet", "WAS", "CALLip", "LCSNet", "LipFormer\n(con landmarks)"]
    unseen = [17.5, 14.6, np.nan, 11.6, 9.64]
    overlap = [4.8, 3.0, 2.48, 2.3, 1.45]
    x = np.arange(len(metodos)); w = 0.38
    for off, vals, lab, c in ((-w / 2, unseen, "Hablantes nuevos", MORADO), (w / 2, overlap, "Hablantes ya vistos", GRIS)):
        for xi, v in zip(x + off, vals):
            if np.isnan(v):
                axs[0].text(xi, 0.4, "sin dato", rotation=90, ha="center", fontsize=8, color="#6b6880")
        axs[0].bar(x + off, np.nan_to_num(vals), w, color=c, label=lab)
        for xi, v in zip(x + off, vals):
            if not np.isnan(v):
                axs[0].text(xi, v + 0.25, f"{v:g}", ha="center", fontsize=8.5)
    axs[0].patches[4].set_color(TEAL)
    axs[0].set_xticks(x); axs[0].set_xticklabels(metodos, fontsize=9)
    axs[0].set_ylabel("WER (%)  (menor es mejor)")
    axs[0].set_title("GRID (ingles)", loc="left", fontweight="bold")
    axs[0].set_ylim(0, 21)
    axs[0].legend(frameon=False, loc="upper right")

    m2 = ["LipNet", "CSSMCM", "LCSNet", "LipFormer\n(con landmarks)"]
    u2 = [52.18, 50.08, 46.98, 43.18]
    o2 = [33.41, 32.48, 30.03, 27.79]
    x2 = np.arange(len(m2))
    b1 = axs[1].bar(x2 - w / 2, u2, w, color=MORADO, label="Hablantes nuevos")
    axs[1].bar(x2 + w / 2, o2, w, color=GRIS, label="Hablantes ya vistos")
    b1[3].set_color(TEAL)
    for xi, v in zip(x2 - w / 2, u2):
        axs[1].text(xi, v + 0.8, f"{v:g}", ha="center", fontsize=8.5)
    for xi, v in zip(x2 + w / 2, o2):
        axs[1].text(xi, v + 0.8, f"{v:g}", ha="center", fontsize=8.5)
    axs[1].set_xticks(x2); axs[1].set_xticklabels(m2, fontsize=9)
    axs[1].set_ylabel("CER (%)  (menor es mejor)")
    axs[1].set_title("CMLR (chino)", loc="left", fontweight="bold")
    axs[1].set_ylim(0, 66)
    axs[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Lectura de labios: usar landmarks ayuda con personas que el modelo no vio", fontweight="bold", fontsize=14, x=0.01, ha="left")
    guardar(fig, "08_lipformer_landmarks_en_lectura_de_labios.png",
            "Xue et al., LipFormer, Tablas V y IV (pag. 7). Son frases completas en otros idiomas, no palabras sueltas: sirve como contexto, no como numero comparable con tu 70%.")


# ===================================================================== 6
# Tus numeros reales (outputs/reporte_resultados.md y entrenamiento del clasificador de audio)
def g_tus_modelos():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [4, 1.3]})
    n = ["Top-1\n(clase exacta, 58)", "Top-3", "Bien o mal\n(binaria)", "Confianza\nsuficiente"]
    v = [67.17, 92.46, 75.98, 87.17]
    axs[0].bar(n, v, color=TEAL, width=0.55)
    for i, vi in enumerate(v):
        axs[0].text(i, vi + 1.2, f"{vi:.2f}%", ha="center", fontweight="bold")
    axs[0].axhline(100 / 58, color=NARANJA, ls="--", lw=1.4)
    axs[0].text(3.45, 100 / 58 + 1.5, "azar = 1.7%", color=NARANJA, ha="right", fontsize=9)
    axs[0].set_ylim(0, 108); axs[0].set_ylabel("%")
    axs[0].set_title("Modelo de video (TCN + Conformer)\n3064 clips de validacion - 17.18 ms por prediccion", loc="left", fontweight="bold", fontsize=11.5)
    axs[1].bar(["Audio\n(wav2vec2 + cabeza)"], [97.40], color=MORADO, width=0.55)
    axs[1].text(0, 97.40 + 1.2, "97.40%", ha="center", fontweight="bold")
    axs[1].set_ylim(0, 108)
    axs[1].set_title("Modelo de audio\n3004 clips, 58 clases", loc="left", fontweight="bold", fontsize=11.5)
    guardar(fig, "09_tus_modelos_resultados.png",
            "outputs/reporte_resultados.md (video) y entrenamiento del clasificador de audio. Mismo conjunto de voces en train y validacion: ver limitaciones.")


if __name__ == "__main__":
    g_wav2vec_librispeech(); g_wav2vec_pocos_datos(); g_timit()
    g_conformer_params(); g_conformer_ablacion(); g_tcn()
    g_hrnet(); g_lipformer(); g_tus_modelos()
