#!/bin/bash
# Link host data into the places pyVideoTrans already reads and writes.
set -euo pipefail

mkdir -p /data/in /data/out /data/models /data/logs /data/tmp /data/f5-tts /data/config

cd /app
if [[ $# -eq 0 ]]; then
    exec python3 webui.py --host 0.0.0.0 --port 7860
fi
exec "$@"
