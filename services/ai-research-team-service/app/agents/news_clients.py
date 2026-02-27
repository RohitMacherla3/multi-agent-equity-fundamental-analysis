from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import settings


@dataclass
class NewsFetchResult:
    provider: str
    rows: list[dict[str, Any]]
    error: str | None = None


class TavilyNewsClient:
    def __init__(self) -> None:
        self.api_key = settings.tavily_api_key
        self.timeout_sec = settings.news_timeout_sec
        self.max_results = max(1, min(settings.news_max_results, 10))
        self.recency_days = max(1, settings.news_recency_days)
        self.endpoint = "https://api.tavily.com/search"

    def fetch(self, query: str, ticker: str) -> NewsFetchResult:
        if not self.api_key:
            return NewsFetchResult(provider="tavily", rows=[], error="tavily_api_key_missing")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "topic": "news",
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "days": self.recency_days,
        }

        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                response = client.post(self.endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            return NewsFetchResult(provider="tavily", rows=[], error=f"tavily_request_failed:{exc}")

        items = data.get("results", []) if isinstance(data, dict) else []
        rows: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()
        for idx, item in enumerate(items[: self.max_results]):
            title = str(item.get("title") or "")
            url = str(item.get("url") or "")
            content = str(item.get("content") or "")
            published = str(item.get("published_date") or "")
            source = str(item.get("source") or "")
            preview = f"{title}. {content}".strip()[:260]
            rows.append(
                {
                    "chunk_id": f"news-{ticker}-{idx}",
                    "accession_no": f"NEWS:{ticker}:{idx}",
                    "ticker": ticker,
                    "form_type": "NEWS",
                    "section_name": source or "NEWS",
                    "chunk_index": idx,
                    "text_preview": preview,
                    "source_url": url,
                    "title": title,
                    "published_at": published,
                    "created_at": now,
                    "score": max(0.45, 0.72 - (idx * 0.05)),
                    "source_type": "news",
                }
            )

        return NewsFetchResult(provider="tavily", rows=rows)

