#!/usr/bin/env bash
# Start pyVideoTrans in a rootless Podman container with ROCm.
# Video, models, logs and settings stay in ./data on the host.
# The container is not installed as a service and does not start at boot.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${PVT_IMAGE:-localhost/pyvideotrans-rocm}"
NAME="${PVT_NAME:-pyvideotrans}"
PORT="${PVT_PORT:-7860}"

# Device 0 is the RX 9060 XT (gfx1200). Device 1 is the Raphael iGPU.
HIP_DEVICE="${PVT_HIP_DEVICE:-0}"

ensure_dirs() {
    mkdir -p \
        "$ROOT/data/in" \
        "$ROOT/data/out" \
        "$ROOT/data/models" \
        "$ROOT/data/logs" \
        "$ROOT/data/tmp" \
        "$ROOT/data/tmp/numba" \
        "$ROOT/data/tmp/miopen" \
        "$ROOT/data/tmp/cache" \
        "$ROOT/data/home" \
        "$ROOT/data/f5-tts" \
        "$ROOT/data/config"
}

build_image() {
    echo "Building $IMAGE"
    podman build -f "$ROOT/Containerfile" -t "$IMAGE" "$ROOT"
}

running() {
    podman container exists "$NAME" \
        && [[ "$(podman inspect -f '{{.State.Running}}' "$NAME")" == "true" ]]
}

run_args=(
    --name "$NAME"
    --userns=keep-id
    --security-opt=label=disable
    --security-opt=seccomp=unconfined
    --device=/dev/kfd
    --device=/dev/dri
    --ipc=host
    --cap-add=SYS_PTRACE
    --ulimit=memlock=-1
    --ulimit=stack=67108864
    -e "HIP_VISIBLE_DEVICES=${HIP_DEVICE}"
    -e "HOME=/data/home"
    -e "XDG_CACHE_HOME=/data/tmp/cache"
    -e "NUMBA_CACHE_DIR=/data/tmp/numba"
    -e "MIOPEN_USER_DB_PATH=/data/tmp/miopen"
    -e "MIOPEN_CUSTOM_CACHE_DIRECTORY=/data/tmp/miopen"
    -p "127.0.0.1:${PORT}:7860"
    -v "$ROOT/data:/data:z"
)

start() {
    ensure_dirs
    if ! podman image exists "$IMAGE"; then
        build_image
    fi
    if running; then
        echo "pyVideoTrans is already running at http://127.0.0.1:${PORT}"
        return
    fi
    if podman container exists "$NAME"; then
        podman start "$NAME"
    else
        podman run -d "${run_args[@]}" "$IMAGE"
    fi
    echo "pyVideoTrans WebUI is starting at http://127.0.0.1:${PORT}"
    echo "Put videos in $ROOT/data/in. Finished files land in $ROOT/data/out."
    echo "Inside the container those paths are /data/in and /data/out."
    echo "Stop it with: $ROOT/pyvideotrans.sh stop"
}

stop() {
    if podman container exists "$NAME"; then
        podman stop "$NAME"
    else
        echo "pyVideoTrans is not running."
    fi
}

status() {
    if ! podman container exists "$NAME"; then
        echo "stopped (no container)"
        return
    fi
    podman inspect -f '{{.State.Status}}' "$NAME"
}

logs() {
    podman logs -f "$NAME"
}

shell() {
    ensure_dirs
    if running; then
        podman exec -it -w /app "$NAME" bash
        return
    fi
    podman run --rm -it "${run_args[@]}" --entrypoint bash "$IMAGE"
}

# One-shot command with the same mounts. Example:
#   ./pyvideotrans.sh run cli.py --task stt --name /data/in/clip.mp4 --model_name small
run_cmd() {
    ensure_dirs
    if ! podman image exists "$IMAGE"; then
        build_image
    fi
    local tty_args=()
    if [[ -t 0 && -t 1 ]]; then
        tty_args=(-it)
    fi
    podman run --rm "${tty_args[@]}" \
        --userns=keep-id \
        --security-opt=label=disable \
        --security-opt=seccomp=unconfined \
        --device=/dev/kfd \
        --device=/dev/dri \
        --ipc=host \
        --cap-add=SYS_PTRACE \
        --ulimit=memlock=-1 \
        --ulimit=stack=67108864 \
        -e "HIP_VISIBLE_DEVICES=${HIP_DEVICE}" \
        -e "HOME=/data/home" \
        -e "XDG_CACHE_HOME=/data/tmp/cache" \
        -e "NUMBA_CACHE_DIR=/data/tmp/numba" \
        -e "MIOPEN_USER_DB_PATH=/data/tmp/miopen" \
        -e "MIOPEN_CUSTOM_CACHE_DIRECTORY=/data/tmp/miopen" \
        -v "$ROOT/data:/data:z" \
        "$IMAGE" "$@"
}

rebuild() {
    build_image
    if podman container exists "$NAME"; then
        podman rm -f "$NAME"
    fi
    start
}

usage() {
    cat <<EOF
Usage: $0 {start|stop|status|logs|shell|rebuild|run}

Host folder mounted at /data in the container:
  $ROOT/data/in       videos and audio to process
  $ROOT/data/out      finished videos and subtitles
  $ROOT/data/models   Whisper, Hugging Face and ModelScope files
  $ROOT/data/f5-tts   voice-clone reference audio
  $ROOT/data/config   settings written by the app
  $ROOT/data/logs
  $ROOT/data/tmp

WebUI: http://127.0.0.1:${PORT}
The container is not started at boot.
EOF
}

cmd="${1:-}"
case "$cmd" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    logs) logs ;;
    shell) shell ;;
    rebuild) rebuild ;;
    run) shift; run_cmd "$@" ;;
    *) usage; exit 1 ;;
esac
