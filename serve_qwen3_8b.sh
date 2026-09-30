#!/bin/bash
# Serve the Qwen3-8B encoder as the embedder behind `clm-serve`.
#
# This replaces the old vLLM-based serve_qwen3_8b.sh. On AMD Instinct
# MI50 / MI60 / Radeon VII (gfx906) vLLM's EngineCore segfaults inside
# libamdhip64.so, so we use the CLM transformers-based embedder instead —
# it ships with the same pip package and uses the same OpenAI-compatible
# `/v1/embeddings` endpoint as vLLM did.
#
# LAST-token pooling matches what the reference head was trained against
# (CLM_v0.1-8B.pt was trained on Qwen3-8B last-token pooling from vLLM),
# so `clm-serve` does not need to know the embedder backend changed.
#
# Requires a gfx906-compatible PyTorch wheel (e.g. the
# `mixa3607/pytorch-gfx906:v2.11.0-rocm-6.3.4` Docker image, or any
# ROCm-PyTorch build with `PYTORCH_ROCM_ARCH=gfx906`).
#
# Usage: GPU=0 PORT=8090 ./serve_qwen3_8b.sh
set -u
GPU="${GPU:-0}"
PORT="${PORT:-8090}"
MAXLEN="${MAXLEN:-2048}"
DTYPE="${DTYPE:-float16}"            # gfx906 has no native bf16 — keep fp16
HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.1.0}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx906}"
LOGDIR="${LOGDIR:-$(cd "$(dirname "$0")/.." && pwd)/logs}"
mkdir -p "$LOGDIR"
echo "serving Qwen3-8B transformers-embedder on HIP ${GPU} port ${PORT} (dtype ${DTYPE}, gfx ${HSA_OVERRIDE_GFX_VERSION})"
HIP_VISIBLE_DEVICES="$GPU" \
HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE_GFX_VERSION" \
PYTORCH_ROCM_ARCH="$PYTORCH_ROCM_ARCH" \
exec python -u -m clm.transformers_embedder \
    --host 0.0.0.0 \
    --port "$PORT" \
    --model Qwen/Qwen3-8B \
    --dtype "$DTYPE" \
    --max-tokens "$MAXLEN" \
    --device cuda \
    >> "$LOGDIR/embedder_demo_8b.log" 2>&1