from __future__ import annotations

import hashlib
import json
import os
from typing import List

import requests


DEFAULT_DIM = 32


def _fake_embedding(text: str, dim: int) -> List[float]:
    """
    Deterministic, offline-friendly embedding used when no service is available.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals = list(digest)
    # repeat digest to reach requested dimension
    repeated = (vals * ((dim // len(vals)) + 1))[:dim]
    # map bytes to small float range [-1, 1]
    return [((v / 255.0) * 2.0) - 1.0 for v in repeated]


def get_embedding(text: str) -> List[float]:
    """
    Compute an embedding for the provided text.

    Priority:
      1) External embedding endpoint defined by EMBEDDING_BASE_URL / EMBEDDING_MODEL.
      2) Deterministic offline embedding.

    The returned vector length is stable across calls for consistent downstream logic.
    """
    base_url = os.getenv("EMBEDDING_BASE_URL")
    model = os.getenv("EMBEDDING_MODEL", "default-mini")
    dim = int(os.getenv("EMBEDDING_DIM", DEFAULT_DIM))
    if dim <= 0:
        dim = DEFAULT_DIM

    if base_url:
        try:
            resp = requests.post(
                base_url.rstrip("/") + "/embeddings",
                json={"model": model, "input": text},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            vector = None
            if isinstance(data, dict):
                vector = data.get("embedding") or data.get("data", [{}])[0].get("embedding")  # type: ignore[index]
            if isinstance(vector, list) and all(isinstance(v, (int, float)) for v in vector):
                return [float(v) for v in vector][:dim]
        except Exception:
            # fallback to deterministic embedding when remote service is unavailable
            pass

    return _fake_embedding(text, dim)
