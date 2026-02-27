from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class FilingInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    accession_no: str = Field(..., alias="accessionNo")
    source_url: str = Field(..., alias="sourceUrl")
    ticker: Optional[str] = None
    company_name: Optional[str] = Field(default=None, alias="companyName")
    form_type: Optional[str] = Field(default=None, alias="formType")
    filing_date: Optional[str] = Field(default=None, alias="filingDate")


class IndexBatchRequest(BaseModel):
    filings: List[FilingInput]


class IndexItemResult(BaseModel):
    accession_no: str
    chunk_count: int
    status: str
    error: Optional[str] = None


class IndexBatchResponse(BaseModel):
    indexed: int
    failed: int
    results: List[IndexItemResult]


class ChunkView(BaseModel):
    chunk_id: str
    accession_no: str
    ticker: str
    form_type: str
    section_name: str
    chunk_index: int
    text_preview: str
    created_at: datetime


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[ChunkView]


class AgenticAnalysisRequest(BaseModel):
    query: str = Field(..., min_length=3)
    ticker: Optional[str] = None
    top_k: int = Field(default=8, alias="topK", ge=1, le=50)
    max_evidence: int = Field(default=12, alias="maxEvidence", ge=1, le=25)
    include_trace: bool = Field(default=True, alias="includeTrace")


class AgentEvidence(BaseModel):
    chunk_id: str
    accession_no: str
    ticker: str
    form_type: str
    section_name: str
    chunk_index: int
    text_preview: str
    score: float


class AgentTraceStep(BaseModel):
    agent: str
    output: dict


class AgenticAnalysisResponse(BaseModel):
    query: str
    ticker: Optional[str]
    summary: str
    confidence: str
    confidence_score: float = Field(alias="confidenceScore")
    writer_model: str
    critic_notes: List[str]
    evidence: List[AgentEvidence]
    memo_markdown: str
    token_usage: dict
    estimated_cost_usd: float = Field(alias="estimatedCostUsd")
    trace: List[AgentTraceStep]


class EvalRunRequest(BaseModel):
    dataset_path: str = Field(default="app/eval/datasets/analyst_eval_set.json", alias="datasetPath")
    top_k: int = Field(default=8, alias="topK", ge=1, le=50)
    max_evidence: int = Field(default=12, alias="maxEvidence", ge=1, le=25)
    min_retrieval_hit_rate: float = Field(default=0.6, alias="minRetrievalHitRate", ge=0.0, le=1.0)
    min_citation_coverage: float = Field(default=0.6, alias="minCitationCoverage", ge=0.0, le=1.0)
    max_avg_latency_ms: float = Field(default=5000.0, alias="maxAvgLatencyMs", ge=1.0)
    output_csv_path: str | None = Field(default=None, alias="outputCsvPath")
    include_trace: bool = Field(default=True, alias="includeTrace")
