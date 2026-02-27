from __future__ import annotations

from typing import Protocol
import numpy as np


class EmbeddingProvider(Protocol):
    name: str

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        ...
