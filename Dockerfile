# syntax=docker/dockerfile:1.7
#
# CLM (contrastive-lm) serving image for AMD Instinct MI50 (gfx906):
#   Qwen3-8B encoder via transformers + clm-serve HTTP API.
#
# Build:   docker build -t clm-serve:latest .
#          # Pin the base image for reproducibility:
#          docker build --build-arg BASE_IMAGE=mixa3607/pytorch-gfx906:v2.11.0-rocm-6.3.4 -t clm-serve:0.1 .
#
# Run:     see docker/README.md
#
# Image layout:
#   - Base:    mixa3607/pytorch-gfx906 (ships ROCm 6.3.x + PyTorch 2.11 built for
#              gfx906). NOT mobydick/vLLM — vLLM's EngineCore segfaults inside
#              libamdhip64.so on gfx906; transformers + plain torch works fine.
#   - User:    root (the gfx906 ROCm runtime relies on /dev/kfd, which is
#              only readable by root on most hosts; we stay as root inside
#              the container instead of creating a non-root user).
#   - WORKDIR: /opt/clm  (clm package installed editable).
#   - Volumes: /models (HF cache + reference head), /logs (uvicorn).
#   - Ports:   8090 (transformers-embedder /v1/embeddings), 8700 (clm-serve).
#
# NOTE: gfx906 does NOT support bf16 natively; the encoder must run in fp16
# (default) or fp32. bf16 silently up-casts to fp32 (very slow + 2× VRAM).
#
ARG BASE_IMAGE=mixa3607/pytorch-gfx906:v2.11.0-rocm-6.3.4
FROM ${BASE_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    DO_NOT_TRACK=1 \
    # gfx906 is reported as "gfx10.1" by the upstream ROCm 6.x runtime, which is
    # the magic value most ROCm-PyTorch forks expect. Override at build/run
    # time if your fork requires a different value.
    HSA_OVERRIDE_GFX_VERSION=10.1.0 \
    PYTORCH_ROCM_ARCH=gfx906 \
    # Defuse NVIDIA/CUDA discovery inside the ROCm container — harmless on
    # AMD-only hosts, but stops accidental nvidia-container-toolkit lookups
    # when both runtimes are present (e.g. dual-socket dev boxes).
    NVIDIA_VISIBLE_DEVICES=

# System tools. The base image already has bash; we add curl for the
# healthcheck, tini for proper PID-1 signal handling, ca-certificates for
# HTTPS calls (Qwen3-8B + HF download).
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        tini \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/clm

# Copy only package metadata first so that the (slow) `pip install` step is
# cached across source edits. We do not use `requirements.txt` here because it
# does `pip install -e .` without `--no-deps`, which would re-resolve
# torch/transformers against the wheels already in the base image (and
# possibly downgrade them). See the `RUN pip install` block below.
COPY --chown=root:root pyproject.toml README.md ./
COPY --chown=root:root src ./src

# Install as root. The base image already has PyTorch + transformers deps.
# We deliberately do NOT re-resolve torch/transformers to avoid a wheel
# version mismatch with the gfx906 build of PyTorch. We only need:
#   (a) the CLM package itself (`--no-deps -e .`),
#   (b) the small CLM deps that may be missing in the base image
#       (uvicorn for our two FastAPI services, fastapi >= 0.100,
#       httpx for the examples).
#
# NB: do NOT `pip install --upgrade pip` — the gfx906 base image ships pip
# as a debian package (apt-installed, no RECORD metadata), so `pip install
# --upgrade pip` aborts with "Cannot uninstall pip 24.0, RECORD file not
# found". The system pip (≥ 22) understands `--no-deps -e .`, so we just
# use it as-is.
RUN pip install --no-deps -e . \
    && pip install --no-deps "uvicorn>=0.23" "httpx>=0.25"

# Entrypoint + helper. Re-owned by root so they sit in a system path and can
# be invoked without PATH gymnastics.
COPY --chmod=0755 docker/entrypoint.sh      /usr/local/bin/entrypoint.sh
COPY --chmod=0755 docker/wait_for_url.sh    /usr/local/bin/wait_for_url.sh

# Default paths so users can mount volumes without rebuilding.
# CLM_DEVICE stays as "cuda" — ROCm-PyTorch ≥ 2.4 exports the CUDA API as
# an alias for HIP, so `torch.cuda.is_available()` and `torch.device("cuda")`
# resolve to the MI50/MI60/Radeon VII through /dev/kfd.
ENV HF_HOME=/models/hf \
    HUGGINGFACE_HUB_CACHE=/models/hf/hub \
    TRANSFORMERS_CACHE=/models/hf/hub \
    HF_HUB_CACHE=/models/hf/hub \
    CLM_CKPT_DIR=/models/clm \
    CLM_EMB_URL=http://127.0.0.1:8090/v1/embeddings \
    CLM_EMB_MODEL=qwen3-8b \
    CLM_PORT=8700 \
    CLM_DEVICE=cuda \
    CLM_ACTION_CACHE=0 \
    CLM_EMB_PORT=8090 \
    CLM_EMB_MAX_TOKENS=2048 \
    CLM_EMB_DTYPE=float16 \
    LOGDIR=/logs

# Prepare writable mount points for HF cache, reference head and logs.
RUN mkdir -p /models/hf/hub /models/clm /logs

EXPOSE 8090 8700

# The healthcheck probes the clm-serve /health endpoint, which itself pings
# the transformers-embedder via the embedder client. A healthy container
# means both services are alive.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8700/health || exit 1

# tini forwards SIGTERM correctly to the foreground `clm-serve` process and
# cleans up the backgrounded transformers-embedder on shutdown.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]