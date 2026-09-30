# CLM in Docker on AMD Instinct MI50 / MI60 / Radeon VII (gfx906)

One container that runs the **vLLM Qwen3-8B (pooling) encoder (gfx906 fork) +
`clm-serve`** with GPU access via the AMD kernel driver and `/dev/kfd`. The
FastAPI server and its web playground are exposed on `:8700`; the embedder is
on `:8090`.

```
browser ──► clm-serve  (FastAPI, :8700)   GET /  POST /v1/systemone  POST /v1/rank
                │
                ▼  HTTP
            vLLM Qwen3-8B pooling  (GPU, :8090)   /v1/embeddings
```

The image is built on top of the gfx906/mobydick fork of vLLM
([`aiinfos/vllm-gfx906-mobydick`](https://github.com/v-aleks/vllm-gfx906-mobydick)).
That fork ships ROCm 6.3.4 + PyTorch 2.11.0 built for `gfx906`, the vLLM fork
itself, and the Triton-AMD backend for flash-attention.

---

## Prerequisites

* Linux host with an **AMD Instinct MI50 / MI60 / Radeon VII** (gfx906) and
  the **AMD amdgpu driver + ROCm 6.3.4 runtime** installed. The mobydick
  README's [Mini Install Guide][mobydick] has the exact commands; the short
  version is:
  ```bash
  # 1. amdgpu-install 6.3.4 — driver + ROCm
  wget https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/noble/amdgpu-install_6.3.60304-1_all.deb
  sudo apt install ./amdgpu-install_6.3.60304-1_all.deb
  sudo amdgpu-install --usecase=rocm --rocmrelease=6.3.4
  sudo usermod -aG render,video $USER   # re-login after this
  rocm-smi --showproductname             # must show MI50 / MI60 / Radeon VII

  # 2. add iommu=pt for stable multi-GPU RCCL (single MI50 is fine without it)
  sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt /' /etc/default/grub
  sudo update-grub && sudo reboot
  ```
* Docker **≥ 24** with the standard `docker` CLI (no `nvidia-container-toolkit`).
* Your user must be in the host's `video` and `render` groups; the host must
  expose `/dev/kfd` and `/dev/dri` to the container (see Run below).
* (optional) `curl` and `jq` on the host for the smoke tests below.

[mobydick]: https://github.com/v-aleks/vllm-gfx906-mobydick#mini-install-guide-for-gfx906

Verify the host can see the GPU before touching Docker:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render rocm/pytorch:latest rocm-smi
```

---

## Build

```bash
# from the repo root
docker build -t clm-serve:latest .

# Reproducible: pin the gfx906 base image
docker build \
    --build-arg VLLM_IMAGE=aiinfos/vllm-gfx906-mobydick:latest \
    -t clm-serve:0.1 .
```

The first build downloads `aiinfos/vllm-gfx906-mobydick` (~10 GB) and pulls
the Python deps (~3 GB). Subsequent builds reuse cached layers.

---

## Run

The two heavy artefacts — **Qwen3-8B** (~16 GB FP16, cached under
`/models/hf/hub`) and the **reference projection head** (75 MB, cached under
`/models/clm`) — should live on a **named volume** so they survive container
restarts. The first start downloads Qwen3-8B and may take 5–15 minutes;
later starts are ~30 s.

For **debugging** gfx906 boot failures, mount `/logs` as a bind-mount on the
host (not a named volume) so `vllm.log` and uvicorn output survive container
crashes and can be read after the fact with `tail`, `less`, your editor, or
a one-liner copy command. A named volume like `clm-logs:/logs` is convenient
for day-to-day use but requires a temporary alpine container to inspect:

```bash
# Recommended for gfx906 debugging — logs land on the host.
mkdir -p "$PWD/clm-logs"
docker run --rm -d --name clm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --cap-add=SYS_ADMIN \
    --ipc=host \
    -p 8700:8700 \
    -p 8090:8090 \
    -v "$PWD/clm-logs:/logs" \
    -v clm-models:/models \
    clm-serve:latest
# then on the host, in another shell:
tail -F "$PWD/clm-logs/vllm.log"
ls -la "$PWD/clm-logs/"
```

* `--device=/dev/kfd --device=/dev/dri --group-add video --group-add render
  --cap-add=SYS_ADMIN` — the AMD pass-through of the GPU. **Do NOT pass
  `--gpus all`**; that is the NVIDIA/nvidia-container-toolkit path and will
  silently leave the container with no GPU.
* `--ipc=host` — gives vLLM/PyTorch a real `/dev/shm` (avoids DataLoader
  shared-memory errors on long contexts).
* `-p 8700:8700` — the CLM API + playground.
* `-p 8090:8090` — direct access to the vLLM `/v1/embeddings` endpoint
  (optional; useful for debugging or for clients that bypass `clm-serve`).
* `-v "$PWD/clm-logs:/logs"` — bind-mount so logs survive a container crash.
  Drop this and use `-v clm-logs:/logs` (a named volume) once boot is stable.

To pin a specific GPU when the host has more than one:

```bash
docker run --rm -d --name clm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render --cap-add=SYS_ADMIN \
    --ipc=host \
    -p 8700:8700 -p 8090:8090 \
    -v "$PWD/clm-logs:/logs" \
    -v clm-models:/models \
    -e GPU=0 \
    clm-serve:latest
```

---

## Smoke test

```bash
# 1. Logs — wait until you see "[entrypoint] vLLM is up" then "[clm] POST ..."
docker logs -f clm

# 2. Health (returns {"ok": true, "embedder": true, "models": ["clm-latest", "clm-raw"]})
curl -s http://localhost:8700/health | jq .

# 3. system_one — the typed-question endpoint from the README
curl -s -X POST http://localhost:8700/v1/systemone \
    -H 'Content-Type: application/json' \
    -d '{
      "state": "My package never arrived and nobody replies.",
      "questions": {
        "urgent":    {"type": "noul",   "instructions": "Is this urgent?"},
        "team":      {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"billing":"Charges","support":"Customer issues"}},
        "anger":     {"type": "score",  "instructions": "How angry is the customer?",
                      "criteria": ["Calm","Frustrated","Very angry"]}
      }
    }' | jq .

# 4. rank — best-of-N for free-form candidates
curl -s -X POST http://localhost:8700/v1/rank \
    -H 'Content-Type: application/json' \
    -d '{
      "context":"What causes tides on Earth?",
      "question":"",
      "answers":["The Moon gravitational pull.","Photosynthesis.","Because the Earth is round."]
    }' | jq .

# 5. (optional) talk to the embedder directly
curl -s -X POST http://localhost:8090/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-8b","input":["hello world"]}' \
    | jq '.data[0].embedding | length'    # -> 4096

# 6. (optional) confirm the GPU is bound by the container
docker exec clm rocm-smi
```

Open the playground in a browser: <http://localhost:8700/>.

---

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `GPU` | `0` | HIP device index (`HIP_VISIBLE_DEVICES`) inside the container. |
| `VLLM_PORT` | `8090` | Port the vLLM `/v1/embeddings` server listens on. |
| `VLLM_MAX_MODEL_LEN` | `2048` | Encoder context length. Lower it to free GPU memory. |
| `VLLM_UTIL` | `0.75` | `gpu-memory-utilization` for vLLM (0..1). The default is conservative for gfx906 because vLLM's allocator ignores KFD/amdgpu driver overhead. |
| `VLLM_MAX_NUM_SEQS` | `8` | Concurrent sequences in vLLM. Pool encoders don't benefit from large batches. |
| `VLLM_DTYPE` | `float16` | Encoder dtype. **Use `float16` on gfx906** — `bfloat16` is not native and falls back to `float32` (slow + 2× VRAM). |
| `VLLM_LOGGING_LEVEL` | `INFO` | vLLM log verbosity. Use `WARNING` only after a successful first boot. |
| `VLLM_EXTRA_ARGS` | *(empty)* | Extra flags appended to `vllm serve`, e.g. `--quantization awq_marlot`. |
| `SKIP_VLLM` | `0` | Set to `1` to start `clm-serve` against an external embedder (see below). |
| `CLM_PORT` | `8700` | Port for `clm-serve`. |
| `CLM_HOST` | `0.0.0.0` | Bind address for `clm-serve`. |
| `CLM_API_KEY` | *(empty)* | If set, requests must carry `Authorization: Bearer <key>`. |
| `CLM_CKPT` | *(auto)* | Path to a projection-head checkpoint. If empty, `clm-serve` downloads the reference head into `/models/clm`. |
| `CLM_DEVICE` | `cuda` | `cpu`, or `cuda` (NVIDIA backend AND the AMD ROCm/HIP alias torch exposes on the gfx906 fork). |
| `HSA_OVERRIDE_GFX_VERSION` | `10.1.0` | ROCm virtual GFX version the gfx906 fork expects. Only change if your fork documents a different value. |
| `PYTORCH_ROCM_ARCH` | `gfx906` | ROCm arch list for torch. |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | Toggles the Triton-AMD backend in flash-attention-gfx906. Required. |
| `HF_TOKEN` | *(empty)* | Hugging Face token for private/gated repos. Qwen3-8B is public, so this is rarely needed. |
| `LOGDIR` | `/logs` | Where `vllm.log` and uvicorn logs are written. |

### Example: tune memory on a 32 GB MI50

The defaults assume a single MI50 32 GB. Qwen3-8B FP16 alone uses ~16 GB and
vLLM's KV cache + activations eat the rest, so on a stock MI50 32 GB the arena
cache is disabled by default (`CLM_ACTION_CACHE=0`). If vLLM OOMs at start-up,
drop `VLLM_UTIL` further or shorten `VLLM_MAX_MODEL_LEN`:

```bash
docker run --rm -d --name clm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render --cap-add=SYS_ADMIN \
    --ipc=host -p 8700:8700 -p 8090:8090 \
    -v clm-models:/models -v clm-logs:/logs \
    -e VLLM_UTIL=0.65 -e VLLM_MAX_MODEL_LEN=1024 \
    clm-serve:latest
```

For larger headroom, switch to a quantised encoder:

```bash
-e VLLM_EXTRA_ARGS="--quantization awq_marlot"
```

(Requires an AWQ-quantised Qwen3-8B on the Hub, e.g. `Qwen/Qwen3-8B-AWQ`,
and that the gfx906 fork supports that quantisation method.)

---

## Splitting vLLM into a separate container

If you already have a vLLM pooling server reachable from another host, or want
to scale the encoder horizontally:

```bash
# container A: encoder only (use the mobydick base image directly so you don't
# also pull the FastAPI stack)
docker run --rm -d --name clm-encoder \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render --cap-add=SYS_ADMIN \
    --ipc=host \
    -p 8090:8090 \
    -v clm-models:/models \
    -v clm-logs:/logs \
    -e SKIP_VLLM=0 \
    -e CLM_PORT=8700 \
    clm-serve:latest

# container B: API only, points at A
docker run --rm -d --name clm-api \
    -p 8700:8700 \
    -v clm-models:/models \
    -v clm-logs:/logs \
    -e SKIP_VLLM=1 \
    -e CLM_EMB_URL=http://host.docker.internal:8090/v1/embeddings \
    clm-serve:latest
```

(`host.docker.internal` works out of the box on Docker Desktop and on Linux
since Docker 20.10 with `--add-host=host.docker.internal:host-gateway`.)

---

## Volumes

| Mount point | Contents | Notes |
|---|---|---|
| `/models/hf` | Hugging Face cache (Qwen3-8B weights, snapshots, etc.) | Persist this; first download is ~16 GB. |
| `/models/clm` | Reference projection head (`CLM_v0.1-8B.pt`, 75 MB) | Created on first start by `clm-serve`. |
| `/logs`       | `vllm.log`, uvicorn stdout/stderr                  | Useful for diagnosing start-up issues. |

---

## Troubleshooting

### Inspecting logs after a crash

If the container died before you could run `docker logs clm`, the vLLM log is
still there as long as `/logs` was mounted. With the bind-mount shown in Run:

```bash
tail -200 "$PWD/clm-logs/vllm.log"
```

If you used a named volume (`-v clm-logs:/logs`) and the container is gone,
the volume persists at `/var/lib/docker/volumes/clm-logs/_data/` (Docker
internals — never edit directly). Read it via a one-shot container:

```bash
# list the files
docker run --rm -v clm-logs:/logs alpine:3.20 ls -la /logs/

# copy the latest log out
mkdir -p "$PWD/clm-logs"
docker run --rm -v clm-logs:/logs -v "$PWD/clm-logs:/out" alpine:3.20 \
    sh -c 'cp /logs/vllm.log /out/vllm.log && ls -la /out/'

# tail it without copying
docker run --rm -v clm-logs:/logs alpine:3.20 tail -n 200 /logs/vllm.log
```

**`docker: Error response from daemon: could not select device driver "" with
capabilities: [[gpu]]`.** You forgot the AMD flags and `docker` is interpreting
nothing as "NVIDIA". Re-run with
`--device=/dev/kfd --device=/dev/dri --group-add video --group-add render
--cap-add=SYS_ADMIN` — no `nvidia-container-toolkit` is required.

**`docker exec clm rocm-smi` returns "GPU not detected".** The container was
launched without `/dev/kfd`. Verify on the host:

```bash
ls -l /dev/kfd /dev/dri
# /dev/kfd → must exist (ROCm ≥ 5)
# /dev/dri → must exist (kernel driver)
groups                                   # must include 'video' and 'render'
```

If `/dev/kfd` is missing, your host kernel doesn't have AMD's KFD module; see
the [mobydick install guide][mobydick] for the kernel prerequisites.

**`vllm.log` says `OutOfMemoryError: HIP out of memory.`** Same playbook as
CUDA: lower `VLLM_UTIL` (the default on gfx906 is already 0.75; try `0.65`)
and/or `VLLM_MAX_MODEL_LEN` (e.g. `1024`). On a 32 GB MI50 the defaults
already disable the projection-head cache (`CLM_ACTION_CACHE=0`).

**`vllm.log` hangs for 10+ minutes then aborts with `Engine core initialization
failed. See root cause above.`** — the most common gfx906-specific failure
mode. Run the same command with `docker logs clm -f` and look at lines from
`EngineCore pid=...` (they appear in the parent process's stdout). If you see
nothing between the `EngineCore` NIXL warnings and the abort, vLLM was killed
by a SIGKILL from OOM-killer: check `dmesg | grep -i 'killed process'` on the
host. Two reliable mitigations:
- drop `VLLM_UTIL` to `0.6`,
- drop `VLLM_MAX_NUM_SEQS` to `4`.

**`vllm.log` says `RuntimeError: ... attention ... no kernel ... available`.**
The flash-attention-gfx906 backend failed to register and vLLM has no
fallback. Set `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (the default already
does this; only happens if you override it in your `docker run` command).

**`vllm.log` says `RuntimeError: no kernel image is available for execution
on the device` / `gfx906`.** Your build was compiled for a different GFX
version. Either rebuild the wheels with `PYTORCH_ROCM_ARCH=gfx906`, or set
`HSA_OVERRIDE_GFX_VERSION=10.1.0` (the default already does this — check
that you didn't override it to something else).

**`vllm.log` says `ValueError: ... bfloat16 not supported on gfx906 ...`.**
Make sure `VLLM_DTYPE=float16`. The mobydick fork loads bf16 weights in fp32
unless you explicitly override the dtype, which doubles VRAM and is very
slow. `entrypoint.sh` warns and falls back to float16 if you set
`VLLM_DTYPE` to anything else.

**`vllm.log` says `no flash attention backend is available`.** You forgot
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`. The Dockerfile and `entrypoint.sh`
set this by default; only happens if you override it in your `docker run`
command. (See the previous entry for the more common "Engine core
initialization failed" variant of this same root cause.)

**The container is stuck in `starting` for > 10 minutes.** Qwen3-8B is being
downloaded. Watch progress:
`docker logs -f clm | grep -E '(Downloading|loaded|GB)'`. A warm cache (named
volume) makes subsequent restarts near-instant.

**Healthcheck stays `unhealthy`.** `docker exec clm rocm-smi` should show
the MI50. If it shows nothing, your host kernel doesn't see the card — see
the mobydick install guide.

**I want a non-default projection head.** Mount it read-only and pass
`CLM_CKPT=/models/heads/my.pt`:
```bash
-v $PWD/checkpoints/my.pt:/models/heads/my.pt:ro \
-e CLM_CKPT=/models/heads/my.pt
```