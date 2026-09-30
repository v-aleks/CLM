# syntax=docker/dockerfile:1.7
#
# CLM (contrastive-lm) serving image for AMD Instinct MI50 (gfx906):
#   vLLM Qwen3-8B pooling encoder (from the gfx906/mobydick fork) + clm-serve.
#
# Build:   docker build -t clm-serve:latest .
#          # Pin the base image for reproducibility:
#          docker build --build-arg VLLM_IMAGE=aiinfos/vllm-gfx906-mobydick:latest -t clm-serve:0.1 .
#
# Run:     see docker/README.md
#
# Image layout:
#   - Base:    aiinfos/vllm-gfx906-mobydick (ships ROCm 6.3.4 + PyTorch 2.11
#              built for gfx906 + vLLM fork + flash-attention-gfx906).
#   - User:    non-root `clm` (UID 1000).
#   - WORKDIR: /opt/clm  (clm package installed editable).
#   - Volumes: /models (HF cache + reference head), /logs (vLLM + uvicorn).
#   - Ports:   8090 (vLLM /v1/embeddings), 8700 (clm-serve API + playground).
#
# NOTE: gfx906 does NOT support bf16 natively; vLLM must be launched with
# --dtype float16 or it will load bf16 weights in fp32 (very slow + 2× VRAM).
#
ARG VLLM_IMAGE=aiinfos/vllm-gfx906-mobydick:latest
FROM ${VLLM_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    DO_NOT_TRACK=1 \
    VLLM_LOGGING_LEVEL=WARNING \
    # gfx906 is reported as "gfx10.1" by the upstream ROCm 6.x runtime, which is
    # the magic value most ROCm-PyTorch and vLLM gfx906 forks expect. Override at
    # build/run time if your fork requires a different value.
    HSA_OVERRIDE_GFX_VERSION=10.1.0 \
    PYTORCH_ROCM_ARCH=gfx906 \
    # flash-attention-gfx906 uses a Triton-AMD backend; the fork's flash-attn
    # wrapper silently no-ops without this flag, which then crashes vLLM with
    # "no attention backend available" on long contexts.
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    # Defuse NVIDIA/CUDA discovery inside the ROCm container — harmless on
    # AMD-only hosts, but stops accidental nvidia-container-toolkit lookups
    # when both runtimes are present (e.g. dual-socket dev boxes).
    NVIDIA_VISIBLE_DEVICES=

# System tools. The base image already has bash, curl is added for the
# healthcheck, tini for proper PID-1 signal handling, ca-certificates for
# HTTPS calls (Qwen3-8B + HF download).
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        tini \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. UID/GID 1000 to match the typical host user; easy to override
# at runtime via `--user $(id -u):$(id -g)` if needed.
RUN groupadd -g 1000 clm 2>/dev/null || true \
    && useradd  -u 1000 -g clm -m -s /bin/bash clm 2>/dev/null || true

WORKDIR /opt/clm

# Copy only package metadata first so that the (slow) `pip install` step is
# cached across source edits. We do not use `requirements.txt` here because it
# does `pip install -e .` without `--no-deps`, which would re-resolve
# torch/vllm against the wheels already in the base image (and possibly
# downgrade them). See the `RUN pip install` block below.
COPY --chown=clm:clm pyproject.toml README.md ./
COPY --chown=clm:clm src ./src

# Install as the non-root user. We deliberately do NOT resolve CLM's runtime
# dependencies (torch / vllm / fastapi / numpy / requests) — they are already
# installed in the base image at versions compatible with
# `aiinfos/vllm-gfx906-mobydick` (PyTorch 2.11.0 + ROCm 6.3.4 + vLLM fork),
# and re-resolving them risks a torch downgrade or a vLLM wheel mismatch. We
# only need to (a) install the CLM package itself (`--no-deps -e .`),
# (b) install the CLM deps that are NOT in the base image (`uvicorn`; vLLM's
# `fastapi[standard]` ≥ 0.133 satisfies `fastapi>=0.100`), and (c) the small
# extras the examples need (`httpx`).
USER clm
RUN pip install --upgrade pip \
    && pip install --no-deps -e . \
    && pip install "uvicorn>=0.23" "httpx>=0.25"

# Entrypoint + helper. Re-owned by root so they sit in a system path and can
# be invoked without PATH gymnastics.
USER root
COPY --chmod=0755 docker/entrypoint.sh      /usr/local/bin/entrypoint.sh
COPY --chmod=0755 docker/wait_for_url.sh    /usr/local/bin/wait_for_url.sh

# Default paths so users can mount volumes without rebuilding.
# CLM_DEVICE stays as "cuda" — ROCm-PyTorch ≥ 2.4 exports the CUDA API as an
# alias for HIP, so `torch.cuda.is_available()` and `torch.device("cuda")`
# resolve to the MI50/MI60/Radeon VII through /dev/kfd. CLM_DEVICE=rocm would
# be invalid; users override only when they want CPU.
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
    VLLM_PORT=8090 \
    VLLM_MAX_MODEL_LEN=2048 \
    VLLM_UTIL=0.85 \
    VLLM_MAX_NUM_SEQS=32 \
    VLLM_DTYPE=float16 \
    GPU=0

# Prepare writable mount points for HF cache, reference head and logs.
RUN mkdir -p /models/hf/hub /models/clm /logs \
    && chown -R clm:clm /models /logs

USER clm

EXPOSE 8090 8700

# The healthcheck probes the clm-serve /health endpoint, which itself pings
# vLLM /v1/models via the embedder client. So a healthy container means both
# services are alive.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8700/health || exit 1

# tini forwards SIGTERM correctly to the foreground `clm-serve` process and
# cleans up the backgrounded vLLM child on shutdown.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]