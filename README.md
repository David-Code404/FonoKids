# FonoKids

Aplicación web para que niños practiquen la pronunciación de sonidos difíciles del español (C/K, G suave, sinfones, S y RR fuerte). Evalúa la voz y el movimiento de la cara con modelos de IA.

## Cómo funciona

- **Audio:** wav2vec2 (fonemas) con puntuación de pronunciación (GOP) y un clasificador de audio entrenado (58 clases).
- **Video:** modelo de cara completa (68 puntos de referencia faciales, TCN + Conformer, 54 clases). Se usa solo la cara completa; el modelo de solo labios fue retirado.
- **Decisión:** el audio decide cuando el clasificador es seguro; el video solo bloquea cuando está seguro. Si una palabra no tiene modelo de video (por ejemplo `perro`), se evalúa solo con audio.
- **Ejercicios de calentamiento:** praxias con cámara (sonrisa, beso, boca abierta, mejillas), soplo con micrófono y ejercicios de voz (pa-ta-ka y trabalenguas). Se estima el esfuerzo en estrellas.

## Estructura

- `server/`: API en FastAPI (`main.py`, base de datos).
- `web/`: aplicación React + Vite.
- `scripts/`: entrenamiento y utilidades de audio.
- `Colab/`: cuadernos para extraer landmarks y entrenar el modelo de cara completa.
- `documentacion/`: documento Word, presentación, artículo, HTML de estudio del pipeline.
- `base de datos/`: esquema SQL.
- `data/` y `models/`: datos y modelos entrenados (no se versionan).

## Puesta en marcha

Servidor (Python):

```bash
pip install -r requirements.txt
python server/main.py
```

El servidor queda en `http://127.0.0.1:8000`. Necesita en local `models/audio_classifier.pth` y `data/modelos_cara_completa/best_cara_completa.pth`.

Aplicación web:

```bash
cd web
npm install
npm run dev
```

Pruebas de la estimación de esfuerzo:

```bash
cd web
npm test
```

## Entrenamiento

El modelo de cara completa se entrena con los cuadernos de `Colab/`: primero `Extraer_Cara_Completa_FonoKids.ipynb` y luego `Entrenar_Cara_Completa_FonoKids.ipynb` (50 épocas, 80/20 estratificado, semilla 42). El clasificador de audio se entrena con `scripts/train_audio_classifier.py`.

## Estado

Versión de prueba. Los umbrales de los ejercicios deben calibrarse con más niños y el contenido debe revisarlo un especialista. El modelo de cara completa todavía necesita validarse con grabaciones reales de la aplicación.
