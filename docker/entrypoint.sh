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
#   VLLM_UTIL                 - gpu-memory-utilization for vLLM, 0..1 (default: 0.55)
#   VLLM_MAX_NUM_SEQS         - max concurrent sequences in vLLM (default: 4)
#   VLLM_DTYPE                - encoder dtype; MUST be float16 on gfx906 (default: float16)
#   VLLM_LOGGING_LEVEL        - log verbosity for vLLM (default: INFO; WARNING hides EngineCore errors)
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
#   NCCL_DEBUG                - NCCL/RCCL log verbosity (default: INFO; flip to WARN after a clean first boot)
#   VLLM_READY_TIMEOUT        - seconds to wait for vLLM /v1/models (default: 300).
#                               Raise to 900–1500 once gfx906 cold-load is known to work.
#   LOGDIR                    - directory for vLLM logs (default: /logs)
#   HF_TOKEN                  - Hugging Face token for gated/private repos (optional)

set -euo pipefail

: "${GPU:=0}"
: "${VLLM_PORT:=8090}"
: "${VLLM_MAX_MODEL_LEN:=2048}"
# The 32 GB MI50 is tight even for Qwen3-8B FP16 alone (16 GB weights).
# vLLM's gpu-memory-utilization allocates a contiguous block up-front; if
# it leaves less than ~6 GB for amdgpu / KFD page tables + PyTorch allocator
# slack + the (independent) EngineCore subprocess, EngineCore dies silently
# after `init_process_group` with no traceback and APIServer reports "Engine
# core initialization failed". 0.55 is the highest value that worked on a
# single MI50 in our tests.
: "${VLLM_UTIL:=0.55}"
: "${VLLM_MAX_NUM_SEQS:=4}"           # pool encoder is small; small batches ease KV-cache pressure
: "${VLLM_DTYPE:=float16}"
: "${VLLM_LOGGING_LEVEL:=INFO}"       # EngineCore FATAL/ERROR must surface; WARNING hides them
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
: "${NCCL_DEBUG:=INFO}"    # INFO shows which transport NCCL chose; flip to WARN after a clean first boot
# Cold-load Qwen3-8B FP16 on gfx906 takes ~8–15 minutes; for iterative
# debugging we default to 5 minutes so a broken build fails fast. Override
# with `-e VLLM_READY_TIMEOUT=1500` (or any value) once you're confident
# vLLM eventually boots.
: "${VLLM_READY_TIMEOUT:=300}"
: "${LOGDIR:=/logs}"

mkdir -p "$LOGDIR"
echo "[entrypoint] HF_HOME=${HF_HOME:-/models/hf}  CLM_CKPT_DIR=${CLM_CKPT_DIR:-/models/clm}"
echo "[entrypoint] ROCm gfx=${HSA_OVERRIDE_GFX_VERSION}  PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}  flash-triton=${FLASH_ATTENTION_TRITON_AMD_ENABLE}"

start_vllm() {
  echo "[entrypoint] starting vLLM Qwen3-8B on HIP device ${GPU}, port ${VLLM_PORT}"
  echo "[entrypoint]   max-model-len=${VLLM_MAX_MODEL_LEN}  gpu-mem-util=${VLLM_UTIL}  max-num-seqs=${VLLM_MAX_NUM_SEQS}  dtype=${VLLM_DTYPE}"
  echo "[entrypoint]   logging-level=${VLLM_LOGGING_LEVEL}"
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    echo "[entrypoint]   extra args: ${VLLM_EXTRA_ARGS}"
  fi

  # Surface the host's free VRAM so a silent OOM on gfx906 is easy to spot.
  if command -v rocm-smi >/dev/null 2>&1; then
    echo "[entrypoint] --- rocm-smi ---"
    rocm-smi --showproductname --showmeminfo vram 2>&1 | head -20 || true
    echo "[entrypoint] ---------------"
  fi

  # gfx906 does not implement bf16 — if VLLM_DTYPE is left at bf16 vLLM will silently
  # up-cast weights, doubling VRAM and tanking throughput. Refuse to start in that case.
  if [[ "${VLLM_DTYPE}" != "float16" && "${VLLM_DTYPE}" != "fp16" && "${VLLM_DTYPE}" != "float32" ]]; then
    echo "[entrypoint] WARNING: VLLM_DTYPE=${VLLM_DTYPE} is not supported on gfx906; falling back to float16." >&2
    VLLM_DTYPE="float16"
  fi

  # We deliberately do NOT pass --enable-prefix-caching: pooling-mode encoders cache
  # last-token outputs and the prefix-cache manager allocates a sizeable
  # scratch buffer that's easy to OOM on a single MI50. Re-enable locally if
  # you have headroom via VLLM_EXTRA_ARGS.
  #
  # vLLM v1 always runs `torch.distributed.init_process_group(backend='nccl')`
  # even for single-GPU — RCCL on gfx906 ships with the mobydick image but the
  # defaults try InfiniBand / GPU-direct P2P that don't exist inside Docker,
  # so the rendezvous hangs and EngineCore never replies. The four NCCL_*
  # vars below force a pure TCP/loopback handshake, which works on a single
  # MI50 in a single container. (Multi-GPU rigs will need different settings.)
  #
  # shellcheck disable=SC2086
  HIP_VISIBLE_DEVICES="${GPU}" \
  HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION}" \
  PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}" \
  FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE}" \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  NCCL_SOCKET_IFNAME=lo \
  NCCL_IB_DISABLE=1 \
  NCCL_P2P_DISABLE=1 \
  NCCL_NET_GDR_LEVEL=0 \
  NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
  VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}" \
  vllm serve Qwen/Qwen3-8B \
      --served-model-name qwen3-8b \
      --runner pooling \
      --enforce-eager \
      --dtype "${VLLM_DTYPE}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${VLLM_UTIL}" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
      --port "${VLLM_PORT}" \
      ${VLLM_EXTRA_ARGS} \
      >> "${LOGDIR}/vllm.log" 2>&1 &

  VLLM_PID=$!
  echo "[entrypoint] vLLM PID=${VLLM_PID}; logs -> ${LOGDIR}/vllm.log}"
  trap 'echo "[entrypoint] stopping vLLM (PID=${VLLM_PID})"; kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true' EXIT

  echo "[entrypoint] waiting for vLLM at http://127.0.0.1:${VLLM_PORT}/v1/models (timeout=${VLLM_READY_TIMEOUT}s) ..."
  if ! /usr/local/bin/wait_for_url.sh "http://127.0.0.1:${VLLM_PORT}/v1/models" "${VLLM_READY_TIMEOUT}" 5; then
    echo "[entrypoint] vLLM failed to become ready in ${VLLM_READY_TIMEOUT}s; tail of log:" >&2
    echo "[entrypoint] ---------------- vllm.log (last 1000 lines) ----------------" >&2
    tail -1000 "${LOGDIR}/vllm.log" >&2 || true
    echo "[entryentry] --------------------------------------------------------" >&2
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