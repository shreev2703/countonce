# CountOnce cloud worker (PLAN 14.4: same image as the laptop worker). NOT built or pushed overnight.
# CUDA 12.8 is the first toolkit with sm_120 (Blackwell) support.
# Weights are mounted at /weights (docker run -v /host/weights:/weights:ro), never baked in.
# To bake them anyway, derive an image: FROM this one, then COPY weights/ /weights/.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    COUNTONCE_WEIGHTS=/weights

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin worker \
    && mkdir -p /weights /data && chown worker:worker /data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY countonce ./countonce
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install ".[recon,data]" safetensors

VOLUME ["/weights", "/data"]
USER worker
# Queue worker against a mounted job database.
ENTRYPOINT ["countonce"]
CMD ["worker", "--db", "/data/jobs.db", "--out", "/data/outputs"]
