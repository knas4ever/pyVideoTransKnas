# pyVideoTrans on ROCm for the Radeon RX 9060 XT (gfx1200).
# The base image already has Python 3.10 and torch 2.7.1 built for ROCm,
# including gfx1200. The app's lockfile would otherwise install CUDA torch.
FROM docker.io/rocm/pytorch:rocm7.2.4_ubuntu22.04_py3.10_pytorch_release_2.7.1

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PATH=/opt/rocm/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        fontconfig \
        fonts-dejavu \
        fonts-liberation \
        fonts-noto-cjk \
        git \
        libdbus-1-3 \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        libsndfile1-dev \
        libsm6 \
        libxext6 \
        libxkbcommon-x11-0 \
        libxrender1 \
        python3-dev \
        rubberband-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
COPY podman/filter-reqs.py /tmp/filter-reqs.py

# Keep the image's ROCm torch. Drop the CUDA wheels the lockfile pins.
RUN uv export --no-dev --extra webui --no-emit-project --frozen -o /tmp/requirements.txt \
    && python3 /tmp/filter-reqs.py /tmp/requirements.txt /tmp/reqs-rocm.txt \
    && uv pip install --python /opt/venv/bin/python3 -r /tmp/reqs-rocm.txt \
    && rm -rf /root/.cache/uv /tmp/requirements.txt /tmp/reqs-rocm.txt /tmp/filter-reqs.py \
    && /opt/venv/bin/python3 -c "import PySide6, gradio, torch; print(torch.__version__, torch.version.hip)"

COPY . .
COPY podman/entrypoint.sh /entrypoint.sh
# The process runs as the host user, so these links are created as root at build time.
RUN chmod +x /entrypoint.sh \
    && ln -sfn /data/out /app/output \
    && ln -sfn /data/models /app/models \
    && ln -sfn /data/logs /app/logs \
    && ln -sfn /data/tmp /app/tmp \
    && ln -sfn /data/f5-tts /app/f5-tts \
    && for f in cfg.json params.json languages.json webui_state.json glossary.txt ass.json codec.json; do \
        ln -sfn "/data/config/$f" "/app/videotrans/$f"; \
    done

EXPOSE 7860
ENTRYPOINT ["/entrypoint.sh"]
