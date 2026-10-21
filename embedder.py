"""Corre en un proceso hijo (ProcessPoolExecutor). Al cerrarlo se libera toda la
RAM/VRAM del modelo; el servidor web nunca importa torch."""
import os

import numpy as np

_model = None


def encode(texts: list[str], query: bool) -> np.ndarray:
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = os.environ.get("QILLQA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        # GPU: fp16 (rápido, poca VRAM). CPU: fp32 (bf16 es ~5x más lento en CPU).
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        _model = SentenceTransformer(os.environ.get("QILLQA_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
                                     device=device, model_kwargs={"dtype": dtype})
    kw = {"prompt_name": "query"} if query and "query" in _model.prompts else {}
    v = _model.encode(texts, normalize_embeddings=True, batch_size=8, **kw)
    return np.asarray(v, dtype=np.float32)
