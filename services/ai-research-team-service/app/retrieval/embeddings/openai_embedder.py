from __future__ import annotations

from typing import Any

import httpx
import numpy as np


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        timeout_sec: float = 60.0,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec
        self.base_url = base_url.rstrip("/")

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.post(f"{self.base_url}/embeddings", headers=headers, json=payload)
            resp.raise_for_status()
            body = resp.json()

        data = body.get("data", [])
        if len(data) != len(texts):
            raise RuntimeError("OpenAI embeddings response length mismatch")

        vectors: list[np.ndarray] = []
        for item in data:
            emb = item.get("embedding")
            if not emb:
                raise RuntimeError("OpenAI embeddings response missing embedding")
            vectors.append(np.array(emb, dtype=np.float32))
        return vectors
