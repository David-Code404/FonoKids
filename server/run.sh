#!/usr/bin/env bash
# Supervisor del servidor -- lo vuelve a levantar solo cada vez que se cae,
# sea por el auto-reinicio de /predict (ver server/main.py, se mata el
# proceso a propósito si una predicción se cuelga más de PREDICT_HARD_TIMEOUT_S
# para liberar la VRAM que un hilo trabado dejó atascada) o por cualquier
# otro motivo. Así el servidor "siempre funciona" sin que alguien tenga que
# reiniciarlo a mano cada vez.
cd "$(dirname "$0")/.."
while true; do
  echo "[run.sh] Arrancando server/main.py..."
  python server/main.py
  echo "[run.sh] server/main.py se cortó (código $?) -- reiniciando en 2s..."
  sleep 2
done
