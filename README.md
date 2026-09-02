# SpeakShadow

Lectura de labios con visión por computadora (MediaPipe + ResNet18 + Transformer)
y una app de Flutter para probarlo desde el celular.

## Estructura

- `scripts/` — pipeline de datos y entrenamiento (grabar, extraer landmarks, entrenar, probar por webcam).
- `notebooks/` — notebook de entrenamiento en Colab.
- `models/` — `face_landmarker.task` (MediaPipe) y `mejor_modelo_speakshadow.pt` (el checkpoint entrenado, bajado de Colab).
- `data/` — `sesiones_continuas/` (clips crudos) y `dataset_pt/` (landmarks ya procesados).
- `outputs/` — gráficas que genera `scripts/train.py`.
- `server/` — API HTTP (FastAPI) que expone el modelo entrenado para la app.
- `app/` — app Flutter: graba un video corto y muestra la palabra predicha.

## Cómo correr la demo completa

1. Poné el checkpoint entrenado en `models/mejor_modelo_speakshadow.pt` (bajado de Colab).
2. Levantá el servidor en la PC (misma red WiFi que el celular):
   ```
   python server/main.py
   ```
   Se queda escuchando en `http://<IP-de-tu-PC>:8000`. Fijate tu IP local con `ipconfig`.
3. Abrí la app Flutter (`cd app && flutter run` en un celular o emulador con cámara), tocá el ícono de
   configuración y poné la URL del paso anterior (ej: `http://192.168.1.100:8000`).
4. Tocá el botón para grabar, decí la palabra, tocá de nuevo para cortar y predecir.

La app requiere permiso de cámara (lo pide sola la primera vez) y conexión a la misma red WiFi que la PC.

