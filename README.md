# SpeakShadow

Lectura de labios con visión por computadora (HRNet/FAN + TCN + Conformer)
y una app web hecha en **React** para probarlo desde la misma PC, sin
instalar nada (se abre en el navegador).

## Estructura

- `scripts/` — pipeline de datos y entrenamiento (grabar, extraer landmarks, entrenar, probar por webcam con `probar_modelo.py`).
- `notebooks/` — notebook de entrenamiento en Colab.
- `models/` — `best.pth` (el checkpoint entrenado, bajado de Colab) y `face_landmarker.task` (MediaPipe, solo para el overlay visual).
- `data/` — `sesiones_continuas/` (clips crudos grabados para el dataset).
- `server/` — API HTTP (FastAPI) que expone el modelo entrenado a la app web. Incluye `run.sh` (relanza el proceso si se cae) y `watchdog.sh` (vigilante externo que lo reinicia si se cuelga).
- `web/` — app web en **React + Vite**: graba un video corto desde la cámara del navegador y muestra la palabra predicha. Reemplaza a la vieja app de Flutter (ya no está en el repo).

## Cómo correr la demo completa

1. Poné el checkpoint entrenado en `models/best.pth` (bajado de Colab).
2. Levantá el servidor (Windows o Linux, todo en la misma PC):
   ```
   python server/main.py
   ```
   Se queda escuchando solo en `http://localhost:8000` (no se expone a la red).

   Opcional pero recomendado: para que el servidor se reinicie solo si se cuelga (ej. un driver de GPU que falla intermitentemente), corré en paralelo:
   ```
   bash server/run.sh       # relanza main.py si el proceso muere
   bash server/watchdog.sh  # lo mata y fuerza el reinicio si deja de responder
   ```
   Ambos scripts corren igual en Windows (Git Bash) y en Linux.
3. Levantá la app web (`cd web && npm install && npm run dev`) y abrila en `http://localhost:5173`.
4. Tocá el botón para grabar, decí la palabra, tocá de nuevo para cortar y predecir.

La app requiere permiso de cámara (lo pide el navegador la primera vez).

## Cómo actualizar cada parte

- **Modelo (`models/best.pth`)**: se entrena aparte, en `notebooks/entrenar_landmarks_colab.ipynb` (Colab) o con `scripts/train_landmarks_transformer.py` en local. El checkpoint que genera tiene que incluir `class_names`, `max_frames` y `model_state_dict` (claves obligatorias que lee `load_everything()` en `server/main.py`) -- `input_dim` es opcional, si no viene usa 120 por defecto. Para actualizarlo: reemplazá el archivo `models/best.pth` por el nuevo checkpoint y reiniciá el servidor (`python server/main.py`, o esperá a que `server/run.sh` lo relance si ya estaba corriendo) -- las clases que reconoce `/predict` son las que traiga ESE checkpoint, no hace falta tocar código.
- **Dataset (`data/sesiones_continuas/`)**: cada carpeta es una palabra/frase, con sus clips adentro (`<palabra>/clips/*.avi`). Se agregan grabando con `scripts/grabar_video_continuo.py` (dataset de entrenamiento) o usándolos desde la app web (se guardan solos ahí cuando `/predict` detecta una frase de riesgo real, con persona "web"). Agregar clips a una carpeta NO actualiza el modelo solo -- hay que reentrenar y reemplazar `best.pth` para que el modelo aprenda las palabras nuevas.
- **Backend (`server/`)**: es Python puro (FastAPI + PyTorch + OpenCV + face-alignment). Después de tocar `server/main.py` alcanza con reiniciar el proceso (`Ctrl+C` y volver a correr `python server/main.py`, o dejar que `server/run.sh` lo detecte y lo relance solo). Si cambiás `requirements.txt`, correr `pip install -r requirements.txt` de nuevo.
- **Frontend (`web/`)**: React + Vite. En desarrollo (`npm run dev`) los cambios de código se ven solos (hot reload), sin reiniciar nada. Para dependencias nuevas, `cd web && npm install`. Para una build de producción, `cd web && npm run build` (queda en `web/dist/`).

## Backend en Linux

El servidor (`server/main.py`) es Python puro (FastAPI + PyTorch + OpenCV +
face-alignment) y corre igual en Linux que en Windows. Las únicas partes
específicas de Windows son cosméticas:
- `server/watchdog.sh` usa `taskkill` si está disponible y cae a `kill -9`
  en Linux automáticamente.
- El aviso de compilación de `torch.compile`/Triton en `build_hrnet_detector()`
  es un problema conocido solo en Windows (sin build oficial de Triton) --
  por eso el detector HRNet/FAN se crea con `compile=False`; en Linux esa
  compilación sí podría funcionar, pero se deja desactivada en los dos para
  que el arranque sea siempre rápido y predecible.
