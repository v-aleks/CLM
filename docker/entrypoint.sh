#!/usr/bin/env bash
# Entrypoint for the CLM Docker image on AMD Instinct MI50 (gfx906).
#
# Starts the Qwen3-8B transformers-embedder in the background, waits for it
# to become ready, then execs `clm-serve` in the foreground (PID 1 is `tini`,
# so both processes receive SIGTERM cleanly when the container is stopped).
#
# Required runtime flags on the host:
#   --device=/dev/kfd --device=/dev/dri \
#   --group-add video --group-add render \
#   --cap-add=SYS_ADMIN --ipc=host
# (NOT --gpus all — that is the CUDA/nvidia-container-toolkit path.)
#
# Environment variables (all optional):
#   CLM_EMB_PORT              - port for the transformers-embedder /v1/embeddings (default: 8090)
#   CLM_EMB_MODEL             - HF model id for the encoder (default: Qwen/Qwen3-8B)
#   CLM_EMB_DTYPE             - encoder dtype; MUST be float16 on gfx906 (default: float16)
#   CLM_EMB_MAX_TOKENS        - truncate prompts to this many tokens (default: 2048)
#   SKIP_EMBEDDER             - if "1", skip starting the embedder (use an external embedder URL)
#   CLM_PORT                  - port for the clm-serve FastAPI server (default: 8700)
#   CLM_HOST                  - bind address for clm-serve (default: 0.0.0.0)
#   CLM_API_KEY               - if set, require `Authorization: Bearer <key>` on every request
#   CLM_CKPT                  - path to a projection-head checkpoint; default: download
#   CLM_DEVICE                - device for the projection heads (cpu or "cuda" alias for HIP; default: cuda)
#   HSA_OVERRIDE_GFX_VERSION  - ROCm virtual GFX version (default: 10.1.0)
#   CLM_READY_TIMEOUT         - seconds to wait for embedder /v1/models (default: 600).
#                               Qwen3-8B cold-load on gfx906 takes 5-10 minutes; raise if needed.
#   LOGDIR                    - directory for logs (default: /logs)
#   HF_TOKEN                  - Hugging Face token for gated/private repos (optional)

set -euo pipefail

: "${CLM_EMB_PORT:=8090}"
: "${CLM_EMB_MODEL:=Qwen/Qwen3-8B}"
: "${CLM_EMB_DTYPE:=float16}"
: "${CLM_EMB_MAX_TOKENS:=2048}"
: "${SKIP_EMBEDDER:=0}"
: "${CLM_PORT:=8700}"
: "${CLM_HOST:=0.0.0.0}"
: "${CLM_API_KEY:=}"
: "${CLM_CKPT:=}"
: "${CLM_DEVICE:=cuda}"
: "${HSA_OVERRIDE_GFX_VERSION:=10.1.0}"
: "${CLM_READY_TIMEOUT:=600}"
: "${LOGDIR:=/logs}"

mkdir -p "$LOGDIR"
echo "[entrypoint] HF_HOME=${HF_HOME:-/models/hf}  CLM_CKPT_DIR=${CLM_CKPT_DIR:-/models/clm}"
echo "[entrypoint] ROCm gfx=${HSA_OVERRIDE_GFX_VERSION}"

start_embedder() {
  echo "[entrypoint] starting transformers-embedder on ${CLM_EMB_PORT}, model=${CLM_EMB_MODEL}, dtype=${CLM_EMB_DTYPE}"

  # gfx906 does not implement bf16 — if CLM_EMB_DTYPE is set to bf16 the
  # encoder will silently up-cast weights, doubling VRAM and tanking
  # throughput. Refuse to start in that case.
  case "${CLM_EMB_DTYPE}" in
    float16|fp16|float32|fp32) : ;;
    *)
      echo "[entrypoint] WARNING: CLM_EMB_DTYPE=${CLM_EMB_DTYPE} is not supported on gfx906; falling back to float16." >&2
      CLM_EMB_DTYPE="float16"
      ;;
  esac

  HIP_VISIBLE_DEVICES="${GPU:-0}" \
             HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION}" \
             PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx906}" \
             python -u -m clm.transformers_embedder \
                 --host 0.0.0.0 \
                 --port "${CLM_EMB_PORT}" \
                 --model "${CLM_EMB_MODEL}" \
                 --dtype "${CLM_EMB_DTYPE}" \
                 --max-tokens "${CLM_EMB_MAX_TOKENS}" \
                 --device "${CLM_DEVICE}" \
                 >> "${LOGDIR}/embedder.log" 2>&1 &
  EMBEDDER_PID=$!
  echo "[entrypoint] embedder PID=${EMBEDDER_PID}; logs -> ${LOGDIR}/embedder.log"
  trap 'echo "[entrypoint] stopping embedder (PID=${EMBEDDER_PID})"; kill "${EMBEDDER_PID}" 2>/dev/null || true; wait "${EMBEDDER_PID}" 2>/dev/null || true' EXIT

  echo "[entrypoint] waiting for embedder at http://127.0.0.1:${CLM_EMB_PORT}/v1/models (timeout=${CLM_READY_TIMEOUT}s) ..."
  if ! /usr/local/bin/wait_for_url.sh "http://127.0.0.1:${CLM_EMB_PORT}/v1/models" "${CLM_READY_TIMEOUT}" 5; then
    echo "[entrypoint] embedder failed to become ready in ${CLM_READY_TIMEOUT}s; tail of log:" >&2
    echo "[entrypoint] ---------------- embedder.log (last 500 lines) ----------------" >&2
    tail -500 "${LOGDIR}/embedder.log" >&2 || true
    echo "[entrypoint] --------------------------------------------------------" >&2
    exit 1
  fi
  echo "[entrypoint] embedder is up"
}

if [[ "${SKIP_EMBEDDER}" != "1" ]]; then
  start_embedder
else
  echo "[entrypoint] SKIP_EMBEDDER=1, not starting embedder (CLM_EMB_URL=${CLM_EMB_URL:-})"
fi

echo "[entrypoint] starting clm-serve on ${CLM_HOST}:${CLM_PORT}"
export CLM_API_KEY CLM_DEVICE
if [[ -n "${CLM_CKPT}" ]]; then
  exec clm-serve --host "${CLM_HOST}" --port "${CLM_PORT}" \
      --emb-url "http://127.0.0.1:${CLM_EMB_PORT}/v1/embeddings" \
      --emb-model qwen3-8b \
      --max-tokens "${CLM_EMB_MAX_TOKENS}" \
      --device "${CLM_DEVICE}" \
      --ckpt "${CLM_CKPT}"
else
  exec clm-serve --host "${CLM_HOST}" --port "${CLM_PORT}" \
      --emb-url "http://127.0.0.1:${CLM_EMB_PORT}/v1/embeddings" \
      --emb-model qwen3-8b \
      --max-tokens "${CLM_EMB_MAX_TOKENS}" \
      --device "${CLM_DEVICE}"
fi