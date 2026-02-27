import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv


THIS_FILE = Path(__file__).resolve()
SERVICE_ROOT = THIS_FILE.parents[2]          # services/ai-research-team-service
REPO_ROOT = THIS_FILE.parents[4]             # equities-research-agent

# Load root .env first, then service-specific .env (service overrides root).
load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv(SERVICE_ROOT / ".env", override=True)


def _resolved_path(env_key: str, default_rel_or_abs: str) -> str:
    raw = os.getenv(env_key, default_rel_or_abs).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


class Settings(BaseModel):
    app_name: str = "Multi-Agent Equity Fundamental Analysis - AI Research Team Service"
    app_env: str = os.getenv("APP_ENV", "local")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "MEFAResearchBot/1.0 (rohitmacherla125@gmail.com)")
    data_dir: str = _resolved_path("PY_INDEX_DATA_DIR", "data/python-index")
    chroma_dir: str = _resolved_path("PY_CHROMA_DIR", "data/chroma")
    eval_assets_dir: str = _resolved_path("EVAL_ASSETS_DIR", "data/eval-reports/assets/evaluation")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_timeout_sec: float = float(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
    max_chunk_chars: int = int(os.getenv("MAX_CHUNK_CHARS", "1200"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
    agent_top_k_default: int = int(os.getenv("AGENT_TOP_K_DEFAULT", "8"))
    agent_max_evidence_default: int = int(os.getenv("AGENT_MAX_EVIDENCE_DEFAULT", "12"))
    agent_target_citation_coverage: float = float(os.getenv("AGENT_TARGET_CITATION_COVERAGE", "0.8"))
    agent_writer_retry_limit: int = int(os.getenv("AGENT_WRITER_RETRY_LIMIT", "1"))
    agent_swarm_revision_max: int = int(os.getenv("AGENT_SWARM_REVISION_MAX", "0"))
    agent_swarm_enable_news: bool = os.getenv("AGENT_SWARM_ENABLE_NEWS", "true").lower() == "true"
    agent_swarm_sec_facts_max_items: int = int(os.getenv("AGENT_SWARM_SEC_FACTS_MAX_ITEMS", "6"))
    agent_swarm_news_max_items: int = int(os.getenv("AGENT_SWARM_NEWS_MAX_ITEMS", "6"))
    agent_debug_default: bool = os.getenv("AGENT_DEBUG_DEFAULT", "false").lower() == "true"
    news_provider: str = os.getenv("NEWS_PROVIDER", "tavily")
    news_max_results: int = int(os.getenv("NEWS_MAX_RESULTS", "5"))
    news_recency_days: int = int(os.getenv("NEWS_RECENCY_DAYS", "14"))
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    gdelt_api_url: str = os.getenv("GDELT_API_URL", "https://api.gdeltproject.org/api/v2/doc/doc")
    news_timeout_sec: float = float(os.getenv("NEWS_TIMEOUT_SEC", "15"))
    token_chars_per_token: int = int(os.getenv("TOKEN_CHARS_PER_TOKEN", "4"))
    openai_chat_input_cost_per_1k: float = float(os.getenv("OPENAI_CHAT_INPUT_COST_PER_1K", "0.00015"))
    openai_chat_output_cost_per_1k: float = float(os.getenv("OPENAI_CHAT_OUTPUT_COST_PER_1K", "0.0006"))


settings = Settings()
