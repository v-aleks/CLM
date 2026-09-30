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
#   - User:    root (the gfx906 fork relies on /dev/kfd, which is only readable
#              by root on most hosts; we deliberately stay as root inside the
#              container instead of creating a non-root user — the previous
#              attempt to create `clm:1000` failed because `useradd` is not
#              present in this base image).
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
    # EngineCore initialization failures on gfx906 must be visible. WARNING
    # suppresses them; INFO is the right default for first-boot diagnostics.
    VLLM_LOGGING_LEVEL=INFO \
    # NCCL/RCCL on gfx906 inside Docker without IB / GPU-direct — INFO shows
    # which transport was chosen ("NET/Socket/0") so silent hangs are diagnosable.
    NCCL_DEBUG=INFO \
    # gfx906 is reported as "gfx10.1" by the upstream ROCm 6.x runtime, which is
    # the magic value most ROCm-PyTorch and vLLM gfx906 forks expect. Override at
    # build/run time if your fork requires a different value.
    HSA_OVERRIDE_GFX_VERSION=10.1.0 \
    PYTORCH_ROCM_ARCH=gfx906 \
    # flash-attention-gfx906 uses a Triton-AMD backend; the fork's flash-attn
    # wrapper silently no-ops without this flag, which then crashes vLLM with
    # "no attention backend available" on long contexts.
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    # vLLM v1 always runs `torch.distributed.init_process_group(backend='nccl')`
    # even for single-GPU. RCCL on gfx906 + Docker's default netns doesn't
    # rendezvous unless we force loopback TCP and disable IB / GPU-direct P2P.
    NCCL_SOCKET_IFNAME=lo \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=1 \
    NCCL_NET_GDR_LEVEL=0 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    # Defuse NVIDIA/CUDA discovery inside the ROCm container — harmless on
    # AMD-only hosts, but stops accidental nvidia-container-toolkit lookups
    # when both runtimes are present (e.g. dual-socket dev boxes).
    NVIDIA_VISIBLE_DEVICES=

# We deliberately stay as root for the whole image (see header). The base
# image ships most of what we need; we add:
#   - curl   : for the /health probe
#   - tini   : for proper PID-1 signal handling
#   - ca-certificates : HTTPS to Qwen3-8B / Hugging Face
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
# torch/vllm against the wheels already in the base image (and possibly
# downgrade them). See the `RUN pip install` block below.
COPY pyproject.toml README.md ./
COPY src ./src

# Install as root. We deliberately do NOT resolve CLM's runtime dependencies
# (torch / vllm / fastapi / numpy / requests) — they are already installed in
# the base image at versions compatible with `aiinfos/vllm-gfx906-mobydick`
# (PyTorch 2.11.0 + ROCm 6.3.4 + vLLM fork), and re-resolving them risks a
# torch downgrade or a vLLM wheel mismatch. We only need to (a) install the
# CLM package itself (`--no-deps -e .`), (b) install the CLM deps that are NOT
# in the base image (`uvicorn`; vLLM's `fastapi[standard]` ≥ 0.133 satisfies
# `fastapi>=0.100`), and (c) the small extras the examples need (`httpx`).
#
# NB: do NOT `pip install --upgrade pip` — the gfx906 base image ships pip as
# a debian package (apt-installed, no RECORD metadata), so `pip install
# --upgrade pip` aborts with "Cannot uninstall pip 24.0, RECORD file not
# found". The system pip (≥ 22) understands `--no-deps -e .` and the wheel
# format, so we just use it as-is.
RUN pip install --no-deps -e . \
    && pip install --no-deps "uvicorn>=0.23" "httpx>=0.25"

# Entrypoint + helper. Re-owned by root so they sit in a system path and can
# be invoked without PATH gymnastics.
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
    # 32 GB × 0.75 ≈ 24 GB on MI50: leaves headroom for KFD / amdgpu driver
    # overhead that vLLM's gpu-memory-utilization estimator ignores on gfx906.
    VLLM_UTIL=0.55 \
    VLLM_MAX_NUM_SEQS=4 \
    VLLM_DTYPE=float16 \
    GPU=0

# Prepare writable mount points for HF cache, reference head and logs.
RUN mkdir -p /models/hf/hub /models/clm /logs

EXPOSE 8090 8700

# The healthcheck probes the clm-serve /health endpoint, which itself pings
# vLLM /v1/models via the embedder client. So a healthy container means both
# services are alive.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8700/health || exit 1

# tini forwards SIGTERM correctly to the foreground `clm-serve` process and
# cleans up the backgrounded vLLM child on shutdown.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]