# SpeakShadow

Lectura de labios con visión por computadora (HRNet/FAN + TCN + Conformer)
y una app web hecha en **React** para probarlo desde el celular o la PC,
sin instalar nada (se abre en el navegador).

## Estructura

- `scripts/` — pipeline de datos y entrenamiento (grabar, extraer landmarks, entrenar, probar por webcam con `probar_modelo.py`).
- `notebooks/` — notebook de entrenamiento en Colab.
- `models/` — `best.pth` (el checkpoint entrenado, bajado de Colab) y `face_landmarker.task` (MediaPipe, solo para el overlay visual).
- `data/` — `sesiones_continuas/` (clips crudos grabados para el dataset).
- `server/` — API HTTP (FastAPI) que expone el modelo entrenado a la app web. Incluye `run.sh` (relanza el proceso si se cae) y `watchdog.sh` (vigilante externo que lo reinicia si se cuelga).
- `web/` — app web en **React + Vite**: graba un video corto desde la cámara del navegador y muestra la palabra predicha. Reemplaza a la vieja app de Flutter (ya no está en el repo).

## Cómo correr la demo completa

1. Poné el checkpoint entrenado en `models/best.pth` (bajado de Colab).
2. Levantá el servidor (Windows o Linux, misma red WiFi que el celular si vas a usarlo desde ahí):
   ```
   python server/main.py
   ```
   Se queda escuchando en `http://<IP-de-tu-PC>:8000`. Fijate tu IP local con `ipconfig` (Windows) o `ip addr` (Linux).

   Opcional pero recomendado: para que el servidor se reinicie solo si se cuelga (ej. un driver de GPU que falla intermitentemente), corré en paralelo:
   ```
   bash server/run.sh       # relanza main.py si el proceso muere
   bash server/watchdog.sh  # lo mata y fuerza el reinicio si deja de responder
   ```
   Ambos scripts corren igual en Windows (Git Bash) y en Linux.
3. Levantá la app web (`cd web && npm install && npm run dev`) y abrila en el navegador (PC o celular en la misma red). Desde Configuración en la app podés cambiar la URL del servidor si no es la misma máquina (ej. `http://192.168.1.100:8000`).
4. Tocá el botón para grabar, decí la palabra, tocá de nuevo para cortar y predecir.

La app requiere permiso de cámara (lo pide el navegador la primera vez) y conexión a la misma red WiFi que la PC del servidor.

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
