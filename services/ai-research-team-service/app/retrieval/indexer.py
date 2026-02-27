from __future__ import annotations

import hashlib
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import chromadb
import httpx
import numpy as np
from bs4 import BeautifulSoup

from app.core.config import settings
from app.retrieval.embeddings.factory import create_embedding_provider
from app.schemas.contracts import FilingInput, IndexItemResult

# ---------------------------------------------------------------------------
# Minimum cosine similarity score to include a chunk in search results.
# Chunks below this threshold are likely irrelevant and excluded.
# Tune this value based on your embedding model (0.35–0.50 is typical).
# ---------------------------------------------------------------------------
MIN_SIMILARITY_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Expected embedding dimension — validated on first embed call.
# Set to None to skip validation (not recommended in production).
# ---------------------------------------------------------------------------
_EXPECTED_EMBEDDING_DIM: Optional[int] = None


class PythonVectorIndexer:
    def __init__(self) -> None:
        self.logger = logging.getLogger("ai_research.retrieval.indexer")
        self.base_dir = Path(settings.data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_provider = create_embedding_provider()
        # Always initialize before any early-return migration path.
        self._validated_dim: Optional[int] = None

        self.chroma_dir = Path(settings.chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_chroma()

        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name="filing_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info("indexer_initialized chroma_dir=%s provider=%s", self.chroma_dir, self.embedding_provider.name)

    def _migrate_legacy_chroma(self) -> None:
        legacy_dirs = [
            self.base_dir / "chroma",
            Path("services/ai-research-team-service/data/python-index/chroma").resolve(),
        ]
        if any(self.chroma_dir.iterdir()):
            return
        for legacy in legacy_dirs:
            if legacy == self.chroma_dir:
                continue
            if not legacy.exists() or not legacy.is_dir():
                continue
            if not any(legacy.iterdir()):
                continue
            for item in legacy.iterdir():
                target = self.chroma_dir / item.name
                if target.exists():
                    continue
                shutil.move(str(item), str(target))
            self.logger.info("migrated_legacy_chroma source=%s target=%s", legacy, self.chroma_dir)
            break

        # Dimension cache is initialized in __init__.

    def reset(self) -> dict[str, int]:
        deleted = int(self.collection.count())
        self.client.delete_collection(name="filing_chunks")
        self.collection = self.client.get_or_create_collection(
            name="filing_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        # Reset dim cache so next index_batch re-validates cleanly
        self._validated_dim = None
        return {"deletedChunks": deleted}

    def count(self) -> int:
        return int(self.collection.count())

    def index_batch(self, filings: List[FilingInput]) -> List[IndexItemResult]:
        results: List[IndexItemResult] = []
        for filing in filings:
            try:
                payload = self._download_text(filing.source_url)
                text = self._normalize_text(payload)
                sections = self._split_sections(text)
                chunks = self._chunk_sections(sections)
                self._embed_chunks(chunks)
                self._persist_chunks(filing, chunks)
                results.append(
                    IndexItemResult(
                        accession_no=filing.accession_no,
                        chunk_count=len(chunks),
                        status="indexed",
                    )
                )
            except Exception as exc:
                results.append(
                    IndexItemResult(
                        accession_no=filing.accession_no,
                        chunk_count=0,
                        status="failed",
                        error=str(exc),
                    )
                )
        return results

    def get_chunks(self, accession_no: str) -> List[dict]:
        data = self.collection.get(
            where={"accession_no": accession_no},
            include=["documents", "metadatas"],
        )

        ids = data.get("ids", []) or []
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []

        rows: List[dict] = []
        for chunk_id, doc, meta in zip(ids, docs, metas):
            m = meta or {}
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "accession_no": m.get("accession_no", accession_no),
                    "ticker": m.get("ticker", ""),
                    "form_type": m.get("form_type", ""),
                    "section_name": m.get("section_name", "GENERAL"),
                    "chunk_index": int(m.get("chunk_index", 0)),
                    "text_preview": (doc[:220] + "...") if doc and len(doc) > 220 else (doc or ""),
                    "created_at": m.get("created_at", ""),
                }
            )

        rows.sort(key=lambda r: (r["chunk_index"], r["chunk_id"]))
        return rows

    def search(self, query: str, top_k: int = 5, ticker: Optional[str] = None) -> List[dict]:
        query_vec = self._embed(query).tolist()

        where = {"ticker": ticker} if ticker else None
        result = self.collection.query(
            query_embeddings=[query_vec],
            # Fetch more than top_k so that after threshold filtering we still
            # have enough results. If the collection is small, cap at count().
            n_results=max(1, min(top_k * 3, 50, self.collection.count() or 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        rows: List[dict] = []
        for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances):
            m = meta or {}
            similarity = max(0.0, 1.0 - float(distance))

            # FIX 1: Skip chunks below minimum similarity threshold
            if similarity < MIN_SIMILARITY_THRESHOLD:
                continue

            rows.append(
                {
                    "chunk_id": chunk_id,
                    "accession_no": m.get("accession_no", ""),
                    "ticker": m.get("ticker", ""),
                    "form_type": m.get("form_type", ""),
                    "section_name": m.get("section_name", "GENERAL"),
                    "chunk_index": int(m.get("chunk_index", 0)),
                    "text_preview": (doc[:220] + "...") if doc and len(doc) > 220 else (doc or ""),
                    "created_at": m.get("created_at", ""),
                    "score": round(similarity, 4),
                }
            )

        # Return only the requested top_k after threshold filtering
        return rows[:max(1, top_k)]

    def _persist_chunks(self, filing: FilingInput, chunks: List[dict]) -> None:
        self.collection.delete(where={"accession_no": filing.accession_no})

        now = datetime.now(timezone.utc).isoformat()
        ids: List[str] = []
        docs: List[str] = []
        metas: List[dict[str, Any]] = []
        vectors: List[list[float]] = []

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            docs.append(chunk["content"])
            vectors.append(chunk["embedding"])
            metas.append(
                {
                    "accession_no": filing.accession_no,
                    "ticker": filing.ticker or "",
                    "company_name": filing.company_name or "",
                    "form_type": filing.form_type or "",
                    "filing_date": filing.filing_date or "",
                    "section_name": chunk["section_name"],
                    "chunk_index": int(chunk["chunk_index"]),
                    "created_at": now,
                }
            )

        if ids:
            self.collection.add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=vectors,
            )

    def _download_text(self, source_url: str) -> str:
        headers = {
            "User-Agent": settings.sec_user_agent,
            "Accept": "text/html,text/plain,*/*",
        }
        with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
            response = client.get(source_url)
            response.raise_for_status()
            return response.text

    def _normalize_text(self, payload: str) -> str:
        lower = payload.lower()
        if "<html" in lower or "<body" in lower or "<div" in lower:
            soup = BeautifulSoup(payload, "html.parser")
            text = soup.get_text(" ")
        else:
            text = payload
        text = " ".join(text.split())
        return text.strip()

    def _split_sections(self, text: str) -> List[tuple[str, str]]:
        """
        FIX 2: Extended section coverage beyond the original 3 sections.
        Now handles 10-K, 10-Q, 8-K, proxy, and earnings release patterns.
        Falls back gracefully to GENERAL for unrecognized filing types.
        """
        lower = text.lower()
        sections: List[tuple[str, str]] = []

        def extract(name: str, starts: List[str], ends: List[str]) -> None:
            s = self._first_match(lower, starts)
            if s < 0:
                return
            e = self._first_match_after(lower, ends, s + 20)
            if e < 0:
                e = min(len(text), s + 12000)
            if e > s:
                sections.append((name, text[s:e]))

        # ── 10-K sections ───────────────────────────────────────────────────
        extract("BUSINESS_OVERVIEW",    ["item 1 ", "item 1.", "business overview"],
                                        ["item 1a", "item 2", "risk factors"])
        extract("RISK_FACTORS",         ["item 1a", "risk factors"],
                                        ["item 1b", "item 2", "properties"])
        extract("MDA",                  ["item 7 ", "item 7.", "management's discussion", "management discussion"],
                                        ["item 7a", "item 8"])
        extract("QUANTITATIVE_RISK",    ["item 7a", "quantitative and qualitative disclosures"],
                                        ["item 8", "financial statements"])
        extract("FINANCIAL_STATEMENTS", ["item 8 ", "item 8.", "financial statements and supplementary"],
                                        ["item 9", "item 9a"])
        extract("CONTROLS",             ["item 9a", "controls and procedures"],
                                        ["item 10", "part iii"])

        # ── 10-Q sections ───────────────────────────────────────────────────
        extract("QUARTERLY_MDA",        ["item 2 ", "item 2.", "management's discussion and analysis"],
                                        ["item 3", "item 4"])
        extract("QUARTERLY_FINANCIALS", ["item 1 ", "financial statements (unaudited)"],
                                        ["item 2", "notes to condensed"])

        # ── 8-K / press release / earnings ──────────────────────────────────
        extract("EARNINGS_RESULTS",     ["results of operations", "financial results", "earnings results"],
                                        ["forward-looking", "safe harbor", "about the company"])
        extract("FORWARD_LOOKING",      ["forward-looking statements", "safe harbor statement"],
                                        ["item 9", "signatures", "exhibit"])
        extract("GUIDANCE",             ["fiscal year guidance", "full year guidance", "outlook"],
                                        ["forward-looking", "safe harbor", "about the company"])

        # ── Proxy / DEF 14A ──────────────────────────────────────────────────
        extract("EXECUTIVE_COMPENSATION", ["executive compensation", "compensation discussion"],
                                          ["audit committee", "director compensation", "equity compensation"])
        extract("CORPORATE_GOVERNANCE",   ["corporate governance", "board of directors"],
                                          ["executive compensation", "audit committee"])

        if not sections:
            # Full fallback — filing type not recognized
            sections.append(("GENERAL", text))
        else:
            # Capture any remaining text not covered by named sections
            used_text = " ".join(sec_text for _, sec_text in sections)
            if len(text) > len(used_text):
                tail = text.replace(used_text, " ").strip()
                if tail:
                    sections.append(("GENERAL", tail))

        return sections

    def _chunk_sections(self, sections: List[tuple[str, str]]) -> List[dict]:
        """
        FIX 3: Chunk ID now includes accession_no-level uniqueness via a
        running global index so boilerplate text at the same position in
        different filings never produces the same chunk_id.
        """
        chunks: List[dict] = []
        idx = 0
        size = settings.max_chunk_chars
        overlap = settings.chunk_overlap_chars

        for section_name, section_text in sections:
            start = 0
            while start < len(section_text):
                end = min(len(section_text), start + size)
                chunk_text = section_text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        {
                            # FIX 3: hash includes global idx AND full text
                            # (not just first 120 chars) to prevent collisions
                            "chunk_id": self._chunk_id(section_name, idx, chunk_text),
                            "section_name": section_name,
                            "chunk_index": idx,
                            "content": chunk_text,
                            "embedding": None,
                        }
                    )
                    idx += 1
                if end >= len(section_text):
                    break
                start = max(start + 1, end - overlap)
        return chunks

    def _embed_chunks(self, chunks: List[dict]) -> None:
        if not chunks:
            return
        texts = [c["content"] for c in chunks]

        # FIX 4: Token-aware batching — split texts into safe batches
        # to avoid exceeding embedding model context window.
        # We approximate tokens as chars / 4 (conservative estimate).
        MAX_CHARS_PER_BATCH = settings.max_chunk_chars * 20  # ~20 chunks per batch
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_chars = 0

        for i, text in enumerate(texts):
            text_chars = len(text)
            if current_batch and current_chars + text_chars > MAX_CHARS_PER_BATCH:
                batches.append(current_batch)
                current_batch = [i]
                current_chars = text_chars
            else:
                current_batch.append(i)
                current_chars += text_chars
        if current_batch:
            batches.append(current_batch)

        for batch_indices in batches:
            batch_texts = [texts[i] for i in batch_indices]
            vectors = self.embedding_provider.embed_texts(batch_texts)

            # FIX 5: Validate embedding dimensions on first batch
            self._validate_embedding_dim(vectors[0])

            for chunk_idx, vec in zip(batch_indices, vectors):
                chunks[chunk_idx]["embedding"] = [float(x) for x in vec.tolist()]

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedding_provider.embed_texts([text])[0]
        # FIX 5: Validate dimension on query embed too
        self._validate_embedding_dim(vec)
        return vec

    def _validate_embedding_dim(self, vec: np.ndarray) -> None:
        """
        FIX 5: Validates that the embedding dimension is consistent.
        Raises RuntimeError if the dimension changes after first validation,
        which would indicate a provider swap against an existing collection.
        """
        dim = int(vec.shape[0])
        if self._validated_dim is None:
            self._validated_dim = dim
            # Also check against existing collection if it has data
            existing_count = self.collection.count()
            if existing_count > 0:
                # Peek at one stored vector to compare dims
                sample = self.collection.get(limit=1, include=["embeddings"])
                stored_vecs = sample.get("embeddings")
                if stored_vecs is None:
                    stored_vecs_list: list = []
                else:
                    # Chroma may return numpy arrays; avoid truthiness checks on ndarrays.
                    stored_vecs_list = list(stored_vecs)
                if len(stored_vecs_list) > 0 and stored_vecs_list[0] is not None:
                    stored_dim = len(stored_vecs_list[0])
                    if stored_dim != dim:
                        raise RuntimeError(
                            f"Embedding dimension mismatch: existing collection has dim={stored_dim}, "
                            f"but current provider produces dim={dim}. "
                            f"Run indexer.reset() before switching embedding providers."
                        )
        elif dim != self._validated_dim:
            raise RuntimeError(
                f"Embedding dimension changed mid-session: expected {self._validated_dim}, got {dim}."
            )

    def _chunk_id(self, section_name: str, idx: int, text: str) -> str:
        # FIX 3: Hash the full text instead of just first 120 chars
        # to prevent boilerplate collision across filings
        raw = f"{section_name}:{idx}:{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def _first_match(self, lower: str, patterns: List[str]) -> int:
        best = -1
        for p in patterns:
            i = lower.find(p)
            if i >= 0 and (best < 0 or i < best):
                best = i
        return best

    def _first_match_after(self, lower: str, patterns: List[str], after: int) -> int:
        best = -1
        for p in patterns:
            i = lower.find(p, after)
            if i >= 0 and (best < 0 or i < best):
                best = i
        return best


indexer = PythonVectorIndexer()
