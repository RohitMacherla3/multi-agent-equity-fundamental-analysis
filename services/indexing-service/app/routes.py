from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
import httpx

from app.settings import settings


router = APIRouter()


def _forward_get(path: str, params: dict | None = None) -> dict:
    url = f"{settings.ai_research_url.rstrip('/')}{path}"
    with httpx.Client(timeout=settings.timeout_sec) as client:
        resp = client.get(url, params=params)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


def _forward_post(path: str, payload: dict) -> dict:
    url = f"{settings.ai_research_url.rstrip('/')}{path}"
    with httpx.Client(timeout=settings.timeout_sec) as client:
        resp = client.post(url, json=payload)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@router.get("/health")
def health() -> dict:
    upstream = _forward_get("/health")
    return {
        "status": "ok",
        "service": "indexing-service",
        "upstream": upstream,
    }


@router.get("/v1/indexing/provider")
def provider() -> dict:
    return _forward_get("/v1/indexing/provider")


@router.get("/v1/indexing/stats")
def stats() -> dict:
    return _forward_get("/v1/indexing/stats")


@router.post("/v1/indexing/reset")
def reset() -> dict:
    return _forward_post("/v1/indexing/reset", {})


@router.post("/v1/indexing/index-batch")
def index_batch(payload: dict) -> dict:
    return _forward_post("/v1/indexing/index-batch", payload)


@router.get("/v1/indexing/chunks")
def chunks(accession_no: str = Query(..., alias="accessionNo")) -> dict:
    return _forward_get("/v1/indexing/chunks", params={"accessionNo": accession_no})


@router.get("/v1/indexing/search")
def search(q: str = Query(...), top_k: int = Query(5, alias="topK"), ticker: str | None = Query(None)) -> dict:
    params = {"q": q, "topK": top_k}
    if ticker:
        params["ticker"] = ticker
    return _forward_get("/v1/indexing/search", params=params)

