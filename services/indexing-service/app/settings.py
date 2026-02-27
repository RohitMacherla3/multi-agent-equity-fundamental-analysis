from __future__ import annotations

import os


class Settings:
    ai_research_url: str = os.getenv("AI_RESEARCH_URL", "http://localhost:8000")
    timeout_sec: float = float(os.getenv("INDEX_PROXY_TIMEOUT_SEC", "300"))


settings = Settings()
