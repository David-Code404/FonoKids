#!/usr/bin/env bash
# Supervisor del servidor -- lo vuelve a levantar solo cada vez que se cae,
# sea porque server/watchdog.sh lo mató a la fuerza (ver ese archivo -- el
# server ahora predice en su hilo principal, así que si se cuelga ya NO
# puede reiniciarse solo desde adentro) o por cualquier otro motivo. Así el
# servidor "siempre funciona" sin que alguien tenga que reiniciarlo a mano.
#
# Corré server/watchdog.sh en paralelo a este script -- run.sh solo
# relanza el proceso cuando MUERE, no detecta por sí solo si se queda
# colgado pero vivo (eso es lo que hace el vigilante).
#
# El PID que usa el vigilante para matar el proceso lo escribe el propio
# server/main.py (os.getpid(), ver el final de ese archivo) -- NO el $! de
# acá, que en git-bash/MSYS puede no coincidir con el PID real de Windows.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p /tmp
while true; do
  echo "[run.sh] Arrancando server/main.py..."
  python -u server/main.py
  echo "[run.sh] server/main.py se cortó (código $?) -- reiniciando en 2s..."
  sleep 2
done
