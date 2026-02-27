from fastapi import APIRouter, HTTPException, Query

from app.agents.engine import get_workflow
from app.eval.runner import eval_runner
from app.retrieval.indexer import indexer
from app.risk.guardrails import sanitize_prompt
from app.schemas.contracts import (
    AgentEvidence,
    AgentTraceStep,
    AgenticAnalysisRequest,
    AgenticAnalysisResponse,
    ChunkView,
    EvalRunRequest,
    IndexBatchRequest,
    IndexBatchResponse,
    SearchResponse,
)

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "embeddingProvider": indexer.embedding_provider.name,
        "agentEngine": "langgraph",
    }


@router.get("/v1/indexing/provider")
def provider() -> dict:
    return {
        "primaryProvider": indexer.embedding_provider.name,
    }


@router.get("/v1/indexing/stats")
def stats() -> dict:
    return {"chunkCount": indexer.count()}


@router.post("/v1/indexing/reset")
def reset_index() -> dict:
    return indexer.reset()


@router.post("/v1/indexing/index-batch", response_model=IndexBatchResponse)
def index_batch(request: IndexBatchRequest) -> IndexBatchResponse:
    results = indexer.index_batch(request.filings)
    failed = sum(1 for r in results if r.status != "indexed")
    return IndexBatchResponse(
        indexed=len(results) - failed,
        failed=failed,
        results=results,
    )


@router.get("/v1/indexing/chunks")
def get_chunks(accession_no: str = Query(..., alias="accessionNo")) -> dict:
    rows = indexer.get_chunks(accession_no)
    return {
        "accessionNo": accession_no,
        "chunkCount": len(rows),
        "chunks": rows,
    }


@router.get("/v1/indexing/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=2),
    top_k: int = Query(5, alias="topK", ge=1, le=50),
    ticker: str | None = Query(default=None),
) -> SearchResponse:
    rows = indexer.search(q, top_k=top_k, ticker=ticker)
    views = [
        ChunkView(
            chunk_id=row["chunk_id"],
            accession_no=row["accession_no"],
            ticker=row["ticker"],
            form_type=row["form_type"],
            section_name=row["section_name"],
            chunk_index=row["chunk_index"],
            text_preview=row["text_preview"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
    return SearchResponse(query=q, top_k=top_k, results=views)


@router.post("/v1/agents/analyze", response_model=AgenticAnalysisResponse)
def analyze(request: AgenticAnalysisRequest) -> AgenticAnalysisResponse:
    guard = sanitize_prompt(request.query)
    if not guard.allowed:
        raise HTTPException(status_code=400, detail={"error": "risk_control_blocked", "reason": guard.blocked_reason})

    result = get_workflow().analyze(
        query=guard.sanitized_text,
        ticker=request.ticker,
        top_k=request.top_k,
        max_evidence=request.max_evidence,
        include_trace=request.include_trace,
    )
    evidence = [
        AgentEvidence(
            chunk_id=row["chunk_id"],
            accession_no=row["accession_no"],
            ticker=row["ticker"],
            form_type=row["form_type"],
            section_name=row["section_name"],
            chunk_index=row["chunk_index"],
            text_preview=row["text_preview"],
            score=float(row["score"]),
        )
        for row in result.evidence
    ]
    trace = [AgentTraceStep(agent=step["agent"], output=step["output"]) for step in result.trace]
    return AgenticAnalysisResponse(
        query=guard.sanitized_text,
        ticker=request.ticker,
        summary=result.summary,
        confidence=result.confidence,
        confidenceScore=result.confidence_score,
        writer_model=result.writer_model,
        critic_notes=result.critic_notes,
        evidence=evidence,
        memo_markdown=result.memo_markdown,
        token_usage=result.token_usage,
        estimatedCostUsd=result.estimated_cost_usd,
        trace=trace,
    )


@router.post("/v1/risk/sanitize")
def sanitize(payload: dict) -> dict:
    text = str(payload.get("query", ""))
    guard = sanitize_prompt(text)
    return {
        "allowed": guard.allowed,
        "blockedReason": guard.blocked_reason,
        "piiTypes": guard.pii_types,
        "sanitizedQuery": guard.sanitized_text,
    }


@router.post("/v1/eval/run")
def run_eval(request: EvalRunRequest) -> dict:
    gate = {
        "min_retrieval_hit_rate": request.min_retrieval_hit_rate,
        "min_citation_coverage": request.min_citation_coverage,
        "max_avg_latency_ms": request.max_avg_latency_ms,
    }
    return eval_runner.run(
        dataset_path=request.dataset_path,
        top_k=request.top_k,
        max_evidence=request.max_evidence,
        quality_gate=gate,
        output_csv_path=request.output_csv_path,
        include_trace=request.include_trace,
    )
