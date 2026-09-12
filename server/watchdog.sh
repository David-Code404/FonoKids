#!/usr/bin/env bash
# Vigilante EXTERNO del servidor -- necesario porque ahora /predict corre
# en el hilo PRINCIPAL del proceso (ver el comentario grande en
# server/main.py sobre por qué), así que si se cuelga, el propio proceso
# ya no puede detectarlo ni reiniciarse solo (el hilo que haría esa
# vigilancia es justo el que está bloqueado). Este script corre APARTE,
# chequea /health desde afuera, y mata el proceso a la fuerza si deja de
# responder por más de KILL_AFTER_S segundos seguidos -- server/run.sh
# (que tiene que estar corriendo en paralelo) lo vuelve a levantar solo
# apenas lo mata.
#
# Uso: correr junto con server/run.sh (en otra terminal/proceso de fondo).
#     bash server/watchdog.sh
cd "$(dirname "$0")/.."

HEALTH_URL="http://127.0.0.1:8000/health"
CHECK_EVERY_S=5
CURL_TIMEOUT_S=5
# Una predicción real (sin cuelgue) tarda hasta ~90s en el peor caso
# (medido en producción) -- durante ESE tiempo /health tampoco responde,
# porque todo corre en el mismo hilo principal. Este umbral tiene que
# quedar CLARAMENTE por encima de eso, para no matar una predicción lenta
# pero sana. 110s deja margen de sobra.
KILL_AFTER_S=110
# Cargar el modelo + HRNet + precalentamiento AL ARRANCAR también bloquea
# /health (mismo motivo: todo en el hilo principal) y puede tardar más que
# KILL_AFTER_S en un arranque en frío (la primera compilación de HRNet es
# lo más lento). Sin este período de gracia, el vigilante mataría el
# server mientras todavía está cargando por primera vez -- un loop de
# reinicios infinito, nunca llegaría a servir ni una request.
STARTUP_GRACE_S=180
# Misma ruta que escribe server/main.py (os.getpid()) -- DENTRO del
# proyecto, no en /tmp, porque el python.exe nativo de Windows no entiende
# esa ruta y crasheaba apenas arrancaba (ver el comentario en main.py).
PID_FILE="$(dirname "$0")/.server.pid"

last_good=$(date +%s)
last_pid=""
startup_deadline=0

echo "[watchdog] Vigilando $HEALTH_URL cada ${CHECK_EVERY_S}s -- mata el server si no responde ${KILL_AFTER_S}s seguidos (con ${STARTUP_GRACE_S}s de gracia al arrancar)."

while true; do
  sleep "$CHECK_EVERY_S"

  # Nuevo PID detectado (arranque o reinicio) -- reiniciar el reloj y darle
  # el período de gracia completo antes de empezar a exigir respuesta.
  if [ -f "$PID_FILE" ]; then
    current_pid=$(cat "$PID_FILE")
    if [ "$current_pid" != "$last_pid" ]; then
      last_pid="$current_pid"
      last_good=$(date +%s)
      startup_deadline=$(( $(date +%s) + STARTUP_GRACE_S ))
      echo "[watchdog] Proceso nuevo detectado (PID $current_pid) -- período de gracia de ${STARTUP_GRACE_S}s."
    fi
  fi

  if curl -s -m "$CURL_TIMEOUT_S" "$HEALTH_URL" 2>/dev/null | grep -q '"status":"ok"'; then
    last_good=$(date +%s)
    continue
  fi

  now=$(date +%s)
  if [ "$now" -lt "$startup_deadline" ]; then
    continue  # todavía en período de gracia de arranque, no contar esto como cuelgue
  fi

  elapsed=$((now - last_good))
  echo "[watchdog] /health sin responder hace ${elapsed}s..."

  if [ "$elapsed" -ge "$KILL_AFTER_S" ]; then
    if [ -n "$last_pid" ]; then
      echo "[watchdog] $elapsed s sin respuesta -- matando proceso colgado (PID $last_pid)..."
      # taskkill en Windows (Git Bash/MSYS) -- en Linux no existe ese
      # comando, ahí se mata con kill -9 directo (PID nativo, sin el lío
      # de traducción MSYS<->Windows que sí hace falta en Windows).
      if command -v taskkill >/dev/null 2>&1; then
        taskkill //F //PID "$last_pid" //T 2>&1
      else
        kill -9 "$last_pid" 2>&1
      fi
    else
      echo "[watchdog] $elapsed s sin respuesta pero no tengo ningún PID todavía -- no puedo matar nada."
    fi
    # Le da tiempo a run.sh a relanzar y al modelo a recargar antes de
    # volver a contar -- si no, seguiría midiendo desde el mismo last_good
    # viejo y mataría el proceso recién levantado a mitad de carga.
    last_good=$(date +%s)
  fi
done
