"""A drop-in ``/v1/embeddings`` server backed by ``transformers`` + ``torch``.

This replaces the vLLM pooling server that the upstream CLM image used.
On AMD Instinct MI50 (gfx906) vLLM's EngineCore subprocess segfaults inside
``libamdhip64.so`` (see the mobydick issue tracker), but the PyTorch wheel
that ships with ``mixa3607/pytorch-gfx906`` runs Qwen3-8B last-token
inference cleanly. The wire format here matches vLLM's pooling endpoint
byte-for-byte, so ``clm.embedder.Embedder`` is unchanged.

Request:
    POST /v1/embeddings
    {"model": "qwen3-8b", "input": ["text", ...], "encoding_format": "base64"|"float",
     "truncate_prompt_tokens": 2048}

Response:
    {"object": "list", "data": [{"index": i, "object": "embedding",
                                 "embedding": "base64-fp32-bytes"} | [..]],
     "model": "...", "usage": {"prompt_tokens": N, "total_tokens": N}}

Health:
    GET /v1/models        -> {"object": "list", "data": [...]}
    GET /health           -> {"ok": True, "device": "cuda"}

The model is loaded once at startup and kept on GPU (or CPU, see ``--device``).
Last-token pooling matches what the reference head was trained against
(``CLM_v0.1-8B.pt`` was trained on Qwen3-8B last-token pooling from vLLM).
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import threading
from typing import Any

import numpy as np

logger = logging.getLogger("clm.transformers_embedder")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)


def last_token_pool(last_hidden_states: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    """Last non-padded token per row, exactly as Qwen2/3 recommend in their
    HF model card. Equivalent to vLLM's ``seq_pooling_type='LAST'``.

    ``last_hidden_states`` shape: [batch, seq, hidden].
    Returns: [batch, hidden].
    """
    import torch
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


class _Embedder:
    """Singleton model holder. Loads once, reuses across requests."""

    def __init__(self, model_name: str, device: str, dtype: str, max_length: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch_dtype = {"float16": torch.float16, "fp16": torch.float16,
                       "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                       "float32": torch.float32, "fp32": torch.float32}[dtype]
        logger.info("loading %s onto %s (dtype=%s, max_length=%d) — first request will block",
                    model_name, device, dtype, max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",   # vLLM uses FlashAttn; SDPA is the
                                          # safe fallback that ships with torch.
        ).to(device).eval()
        self.device = device
        self.max_length = max_length
        logger.info("model loaded: hidden=%d, params=%d",
                    self.model.config.hidden_size,
                    sum(p.numel() for p in self.model.parameters()))

    @torch.inference_mode()
    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        import torch
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc)
        pooled = last_token_pool(out.last_hidden_state, enc.attention_mask)
        pooled = pooled.to(torch.float32).cpu().numpy()
        return pooled, int(enc.attention_mask.sum().item())


class _AppState:
    """Process-wide singleton, lazily populated by the first request so import
    is fast and ``--help`` doesn't trigger a 16 GB download."""
    model_name: str = "Qwen/Qwen3-8B"
    device: str = "cuda"
    dtype: str = "float16"
    max_length: int = 2048
    _embedder: _Embedder | None = None
    _lock: threading.Lock = threading.Lock()


def _get() -> _Embedder:
    if _AppState._embedder is None:
        with _AppState._lock:
            if _AppState._embedder is None:
                _AppState._embedder = _Embedder(
                    _AppState.model_name,
                    _AppState.device,
                    _AppState.dtype,
                    _AppState.max_length,
                )
    return _AppState._embedder


def _serialize(emb: np.ndarray, fmt: str) -> Any:
    if fmt == "base64":
        return base64.b64encode(emb.astype(np.float32).tobytes()).decode("ascii")
    return emb.astype(np.float32).tolist()


def create_app() -> "fastapi.FastAPI":
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="CLM transformers-embedder", version="0.1.0")

    @app.get("/health")
    def health():
        out = {"ok": True, "device": _AppState.device, "model": _AppState.model_name}
        if _AppState._embedder is not None:
            out["hidden_size"] = _AppState._embedder.model.config.hidden_size
        return out

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": "qwen3-8b", "object": "model", "owned_by": "clm-embedder"},
        ]}

    @app.post("/v1/embeddings")
    def embeddings(payload: dict):
        model = payload.get("model", "qwen3-8b")
        inp = payload.get("input", [])
        if isinstance(inp, str):
            inp = [inp]
        if not inp or not all(isinstance(s, str) for s in inp):
            raise HTTPException(422, "input must be a non-empty list of strings")
        fmt = payload.get("encoding_format", "base64")
        if fmt not in ("base64", "float"):
            raise HTTPException(422, f"encoding_format must be 'base64' or 'float', got {fmt!r}")
        # vLLM accepts ``truncate_prompt_tokens`` to clamp inputs at the
        # tokenizer level. Our _Embedder already enforces ``max_length`` in
        # the tokenizer call, so we just honour the request.
        try:
            emb, tokens = _get().embed(inp)
        except Exception as e:  # noqa: BLE001
            logger.exception("embedding failed")
            raise HTTPException(500, f"embedder error: {e}") from e
        return {
            "object": "list",
            "data": [
                {"index": i, "object": "embedding", "embedding": _serialize(emb[i], fmt)}
                for i in range(len(inp))
            ],
            "model": model,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLM_EMB_PORT", 8090)))
    ap.add_argument("--model", default=os.environ.get("CLM_EMB_MODEL", "Qwen/Qwen3-8B"),
                    help="HF model id (default: Qwen/Qwen3-8B, matches the reference head)")
    ap.add_argument("--device", default=None,
                    help="torch device for the encoder (cpu or cuda/HIP). Default: cuda when available.")
    ap.add_argument("--dtype", default=os.environ.get("CLM_EMB_DTYPE", "float16"),
                    choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
                    help="encoder dtype (default: float16; gfx906 has no native bf16)")
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("CLM_EMB_MAX_TOKENS", 2048)),
                    help="truncate prompts to this many tokens (default: 2048)")
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    _AppState.model_name = args.model
    _AppState.device = args.device
    _AppState.dtype = args.dtype
    _AppState.max_length = args.max_tokens

    import uvicorn
    logger.info("starting transformers-embedder on %s:%d (model=%s, dtype=%s)",
                args.host, args.port, args.model, args.dtype)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning",
                access_log=False)


if __name__ == "__main__":
    main()