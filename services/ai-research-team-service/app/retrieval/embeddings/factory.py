from __future__ import annotations

from app.core.config import settings
from app.retrieval.embeddings.base import EmbeddingProvider
from app.retrieval.embeddings.openai_embedder import OpenAIEmbeddingProvider


def create_embedding_provider() -> EmbeddingProvider:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for indexing and retrieval.")
    return OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        timeout_sec=settings.openai_timeout_sec,
        base_url=settings.openai_base_url,
    )
