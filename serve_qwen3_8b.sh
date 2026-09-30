#!/bin/bash
# Serve ONE Qwen3-8B pooling server as the encoder behind `clm-serve` on AMD
# Instinct MI50 / MI60 / Radeon VII (gfx906). Lean settings so it coexists with
# other GPU work: enforce-eager, modest util, short max-model-len (System One
# states are short). LAST-token pooling + prefix cache, same as the precompute,
# so the embeddings match what the head was trained on.
#
# Requires a ROCm 6.3+ runtime with the gfx906 fork installed:
#   https://github.com/v-aleks/vllm-gfx906-mobydick
#
# Usage: GPU=0 PORT=8090 UTIL=0.35 ./serve_qwen3_8b.sh
set -u
GPU="${GPU:-0}"
PORT="${PORT:-8090}"
UTIL="${UTIL:-0.65}"                 # conservative on gfx906; KFD overhead is real
MAXLEN="${MAXLEN:-2048}"
SEQ="${SEQ:-8}"                      # pooling-mode encoder; small batches are fine
DTYPE="${DTYPE:-float16}"            # gfx906 has no native bf16 — keep fp16
HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.1.0}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx906}"
FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"
LOGDIR="${LOGDIR:-$(cd "$(dirname "$0")/.." && pwd)/logs}"
mkdir -p "$LOGDIR"
echo "serving Qwen3-8B pooling on HIP ${GPU} port ${PORT} (util ${UTIL}, dtype ${DTYPE}, gfx ${HSA_OVERRIDE_GFX_VERSION})"
HIP_VISIBLE_DEVICES="$GPU" \
HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE_GFX_VERSION" \
PYTORCH_ROCM_ARCH="$PYTORCH_ROCM_ARCH" \
FLASH_ATTENTION_TRITON_AMD_ENABLE="$FLASH_ATTENTION_TRITON_AMD_ENABLE" \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
exec vllm serve Qwen/Qwen3-8B \
    --served-model-name qwen3-8b \
    --runner pooling \
    --enforce-eager \
    --dtype "$DTYPE" \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$UTIL" \
    --max-num-seqs "$SEQ" \
    --port "$PORT" \
    >> "$LOGDIR/vllm_demo_8b.log" 2>&1