#!/usr/bin/env bash
# Entrypoint for the CLM Docker image on AMD Instinct MI50 (gfx906).
#
# Starts the Qwen3-8B vLLM pooling encoder in the background, waits for it to
# become ready, then execs `clm-serve` in the foreground (PID 1 is `tini`, so
# both processes receive SIGTERM cleanly when the container is stopped).
#
# Required runtime flags on the host:
#   --device=/dev/kfd --device=/dev/dri \
#   --group-add video --group-add render \
#   --cap-add=SYS_ADMIN --ipc=host
# (NOT --gpus all — that is the CUDA/nvidia-container-toolkit path.)
#
# Environment variables (all optional):
#   GPU                       - HIP device index visible inside the container (default: 0)
#   VLLM_PORT                 - port for the vLLM /v1/embeddings server (default: 8090)
#   VLLM_MAX_MODEL_LEN        - max sequence length for the encoder (default: 2048)
#   VLLM_UTIL                 - gpu-memory-utilization for vLLM, 0..1 (default: 0.85)
#   VLLM_MAX_NUM_SEQS         - max concurrent sequences in vLLM (default: 32)
#   VLLM_DTYPE                - encoder dtype; MUST be float16 on gfx906 (default: float16)
#   VLLM_EXTRA_ARGS           - extra args appended to `vllm serve` (e.g. "--quantization awq_marlot")
#   SKIP_VLLM                 - if "1", skip starting vLLM (use an external embedder)
#   CLM_PORT                  - port for the FastAPI server (default: 8700)
#   CLM_HOST                  - bind address for the FastAPI server (default: 0.0.0.0)
#   CLM_API_KEY               - if set, require `Authorization: Bearer <key>` on every request
#   CLM_CKPT                  - path to a projection-head checkpoint; default: download
#   CLM_DEVICE                - device for the heads (cpu or "cuda" alias for HIP; default: cuda)
#   HSA_OVERRIDE_GFX_VERSION  - ROCm virtual GFX version (default: 10.1.0)
#   PYTORCH_ROCM_ARCH         - ROCm arch list for torch (default: gfx906)
#   FLASH_ATTENTION_TRITON_AMD_ENABLE - flash-attn-gfx906 toggle (default: TRUE)
#   LOGDIR                    - directory for vLLM logs (default: /logs)
#   HF_TOKEN                  - Hugging Face token for gated/private repos (optional)

set -euo pipefail

: "${GPU:=0}"
: "${VLLM_PORT:=8090}"
: "${VLLM_MAX_MODEL_LEN:=2048}"
: "${VLLM_UTIL:=0.85}"
: "${VLLM_MAX_NUM_SEQS:=32}"
: "${VLLM_DTYPE:=float16}"
: "${VLLM_EXTRA_ARGS:=}"
: "${SKIP_VLLM:=0}"
: "${CLM_PORT:=8700}"
: "${CLM_HOST:=0.0.0.0}"
: "${CLM_API_KEY:=}"
: "${CLM_CKPT:=}"
: "${CLM_DEVICE:=cuda}"
: "${HSA_OVERRIDE_GFX_VERSION:=10.1.0}"
: "${PYTORCH_ROCM_ARCH:=gfx906}"
: "${FLASH_ATTENTION_TRITON_AMD_ENABLE:=TRUE}"
: "${LOGDIR:=/logs}"

mkdir -p "$LOGDIR"
echo "[entrypoint] HF_HOME=${HF_HOME:-/models/hf}  CLM_CKPT_DIR=${CLM_CKPT_DIR:-/models/clm}"
echo "[entrypoint] ROCm gfx=${HSA_OVERRIDE_GFX_VERSION}  PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}  flash-triton=${FLASH_ATTENTION_TRITON_AMD_ENABLE}"

start_vllm() {
  echo "[entrypoint] starting vLLM Qwen3-8B on HIP device ${GPU}, port ${VLLM_PORT}"
  echo "[entrypoint]   max-model-len=${VLLM_MAX_MODEL_LEN}  gpu-mem-util=${VLLM_UTIL}  max-num-seqs=${VLLM_MAX_NUM_SEQS}  dtype=${VLLM_DTYPE}"
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    echo "[entrypoint]   extra args: ${VLLM_EXTRA_ARGS}"
  fi

  # gfx906 does not implement bf16 — if VLLM_DTYPE is left at bf16 vLLM will silently
  # up-cast weights, doubling VRAM and tanking throughput. Refuse to start in that case.
  if [[ "${VLLM_DTYPE}" != "float16" && "${VLLM_DTYPE}" != "fp16" && "${VLLM_DTYPE}" != "float32" ]]; then
    echo "[entrypoint] WARNING: VLLM_DTYPE=${VLLM_DTYPE} is not supported on gfx906; falling back to float16." >&2
    VLLM_DTYPE="float16"
  fi

  # shellcheck disable=SC2086
  HIP_VISIBLE_DEVICES="${GPU}" \
  HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION}" \
  PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}" \
  FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE}" \
  vllm serve Qwen/Qwen3-8B \
      --served-model-name qwen3-8b \
      --runner pooling \
      --enforce-eager \
      --enable-prefix-caching \
      --dtype "${VLLM_DTYPE}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${VLLM_UTIL}" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
      --port "${VLLM_PORT}" \
      ${VLLM_EXTRA_ARGS} \
      >> "${LOGDIR}/vllm.log" 2>&1 &

  VLLM_PID=$!
  echo "[entrypoint] vLLM PID=${VLLM_PID}; logs -> ${LOGDIR}/vllm.log"
  trap 'echo "[entrypoint] stopping vLLM (PID=${VLLM_PID})"; kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true' EXIT

  echo "[entrypoint] waiting for vLLM at http://127.0.0.1:${VLLM_PORT}/v1/models ..."
  if ! /usr/local/bin/wait_for_url.sh "http://127.0.0.1:${VLLM_PORT}/v1/models" 600 5; then
    echo "[entrypoint] vLLM failed to become ready in 600s; tail of log:" >&2
    tail -200 "${LOGDIR}/vllm.log" >&2 || true
    exit 1
  fi
  echo "[entrypoint] vLLM is up"
}

if [[ "${SKIP_VLLM}" != "1" ]]; then
  start_vllm
else
  echo "[entrypoint] SKIP_VLLM=1, not starting vLLM (CLM_EMB_URL=${CLM_EMB_URL:-})"
fi

echo "[entrypoint] starting clm-serve on ${CLM_HOST}:${CLM_PORT}"
export CLM_API_KEY CLM_DEVICE
if [[ -n "${CLM_CKPT}" ]]; then
  exec clm-serve --host "${CLM_HOST}" --port "${CLM_PORT}" \
      --emb-url "http://127.0.0.1:${VLLM_PORT}/v1/embeddings" \
      --emb-model qwen3-8b \
      --max-tokens "${VLLM_MAX_MODEL_LEN}" \
      --device "${CLM_DEVICE}" \
      --ckpt "${CLM_CKPT}"
else
  exec clm-serve --host "${CLM_HOST}" --port "${CLM_PORT}" \
      --emb-url "http://127.0.0.1:${VLLM_PORT}/v1/embeddings" \
      --emb-model qwen3-8b \
      --max-tokens "${VLLM_MAX_MODEL_LEN}" \
      --device "${CLM_DEVICE}"
fi