"""Runs in a child process (ProcessPoolExecutor). Closing it frees all of the
model's RAM/VRAM; the web server never imports torch."""
import os

import numpy as np

_model = None


def encode(texts: list[str], query: bool) -> np.ndarray:
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = os.environ.get("QILLQA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        # GPU: fp16 (fast, little VRAM). CPU: fp32 (bf16 is ~5x slower on CPU).
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        _model = SentenceTransformer(os.environ.get("QILLQA_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
                                     device=device, model_kwargs={"dtype": dtype})
    kw = {"prompt_name": "query"} if query and "query" in _model.prompts else {}
    v = _model.encode(texts, normalize_embeddings=True, batch_size=8, **kw)
    return np.asarray(v, dtype=np.float32)
