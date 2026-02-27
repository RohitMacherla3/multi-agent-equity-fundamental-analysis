from __future__ import annotations

from functools import lru_cache

@lru_cache(maxsize=1)
def get_workflow():
    from app.agents.langgraph_workflow import workflow as langgraph_workflow
    return langgraph_workflow
