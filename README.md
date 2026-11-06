# FonoKids

App de práctica de pronunciación infantil: lectura de labios con visión por
computadora (FAN + TCN + Conformer) que evalúa si un chico dijo bien o mal
una palabra objetivo, grabando un clip corto desde el navegador -- sin
instalar nada. Incluye además un módulo de análisis de **audio**
(`facebook/wav2vec2-xlsr-53-espeak-cv-ft`) para dar feedback a nivel de
fonema (ej. "la erre te salió mal"), complementario al análisis visual.

## Convención de clases del modelo

Cada palabra objetivo aporta varias clases, todas con este formato:

```
<palabra>_correcto
<palabra>_incorrecto                  (si no se distingue el tipo de error)
<palabra>_incorrecto_<subtipo>        (ej. lambdacismo, dentalizacion, omision, gliding)
```

El nombre de la clase que gana ya dice qué palabra se intentó, si quedó
bien dicha, y (si el checkpoint lo distingue) qué tipo de error específico
fue -- sin necesitar una lista aparte de palabras. Ver `_parse_clase_predicha()`
en `server/main.py`.

## Estructura

- `scripts/` — pipeline de datos y entrenamiento:
  - `grabar_video_continuo.py` — graba clips (video **+ audio**, 16kHz mono) por clase, uno a la vez.
  - `extraer_landmarks_npy.py` — extrae landmarks de labios de los clips grabados.
  - `train_landmarks_transformer.py` — entrena el modelo TCN + Conformer en local.
  - `audio_pronunciation.py` — evalúa pronunciación por FONEMA a partir del audio, con `wav2vec2-xlsr-53-espeak-cv-ft` (no depende de `espeak-ng`, tiene su propio conversor texto→fonemas para español).
  - `probar_modelo.py` — prueba por webcam en vivo (no se toca, referencia de que el pipeline visual funciona).
- `notebooks/` — notebook de entrenamiento en Colab (mismo pipeline que el script local, para entrenar con GPU gratis).
- `models/` — `best.pth` (checkpoint visual entrenado) y `face_landmarker.task` (MediaPipe, solo para el overlay visual en la app web).
- `data/` — `sesiones_continuas/<clase>/clips/` -- una carpeta por CADA clase (`<palabra>_correcto`, `<palabra>_incorrecto_<subtipo>`, etc.), con los `.avi` (video) y `.wav` (audio) de cada toma.
- `server/` — API HTTP (FastAPI) que expone el modelo entrenado a la app web. Incluye `run.sh` (relanza el proceso si se cae) y `watchdog.sh` (vigilante externo que lo reinicia si se cuelga).
- `web/` — app web en **React + Vite**, pensada para chicos (colores vivos, tipografía Baloo 2). Tres pestañas:
  - **Practicar** — graba el intento y muestra si quedó bien o mal, con feedback grande y festivo.
  - **Logros** — progreso por palabra (estrellas, barra de aciertos).
  - **Mi Diario** — historial de intentos por día.

## Cómo correr la demo completa

1. Poné el checkpoint entrenado en `models/best.pth`.
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
4. En la pestaña **Practicar**, decí la palabra objetivo y tocá el botón para grabar.

La app requiere permiso de cámara (lo pide el navegador la primera vez).

## Cómo grabar el dataset

Por cada palabra objetivo, grabás una carpeta por cada clase (correcta + cada
subtipo de error que quieras distinguir):

```
python scripts/grabar_video_continuo.py
```

Te pregunta qué palabra vas a grabar -- ahí tipeás el nombre completo de la
clase (ej. `perro_correcto`, después `perro_incorrecto_lambdacismo`, etc.).
`S` arranca a grabar, `A` corta y guarda (video + audio), `Q` sale. Cada
toma guarda un `.avi` y un `.wav` con el mismo nombre en
`data/sesiones_continuas/<clase>/clips/`.

Después, por cada carpeta:
```
python scripts/extraer_landmarks_npy.py <clase>
```

## Cómo actualizar cada parte

- **Modelo visual (`models/best.pth`)**: se entrena aparte, en `notebooks/entrenar_landmarks_colab.ipynb` (Colab) o con `scripts/train_landmarks_transformer.py` en local. El checkpoint tiene que incluir `class_names`, `max_frames` y `model_state_dict` (claves obligatorias que lee `load_everything()` en `server/main.py`) -- `input_dim` es opcional, default 120. `class_names` tiene que seguir la convención de arriba. Para actualizarlo: reemplazá `models/best.pth` y reiniciá el servidor -- las clases que reconoce `/predict` son las que traiga ESE checkpoint.
- **Dataset (`data/sesiones_continuas/`)**: agregar clips a una carpeta NO actualiza el modelo solo -- hay que reentrenar y reemplazar `best.pth`.
- **Modelo de audio**: no requiere entrenamiento propio (usa el checkpoint pre-entrenado de Facebook tal cual) -- solo hace falta que `scripts/audio_pronunciation.py` tenga la palabra objetivo bien mapeada a fonemas en `texto_a_fonemas()`.
- **Backend (`server/`)**: Python puro (FastAPI + PyTorch + OpenCV + face-alignment). Reiniciar el proceso después de tocar `server/main.py`. Si cambiás `requirements.txt`, correr `pip install -r requirements.txt` de nuevo.
- **Frontend (`web/`)**: React + Vite. En desarrollo (`npm run dev`) los cambios se ven solos (hot reload). Para dependencias nuevas, `cd web && npm install`. Build de producción: `cd web && npm run build` (queda en `web/dist/`).

## Backend en Linux

El servidor (`server/main.py`) es Python puro y corre igual en Linux que en
Windows. Las únicas partes específicas de Windows son cosméticas:
- `server/watchdog.sh` usa `taskkill` si está disponible y cae a `kill -9`
  en Linux automáticamente.
- El detector de landmarks (`face-alignment`) se crea con `compile=False`
  porque `torch.compile` se cuelga en Windows (sin build de Triton); en
  Linux esa compilación podría funcionar, pero se deja desactivada en los
  dos para que el arranque sea siempre rápido y predecible.
