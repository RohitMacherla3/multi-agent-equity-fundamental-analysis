# UML Diagrams: Multi-Agent Equity Fundamental Analysis

This document contains UML diagrams and implementation-oriented UML notes in one place.

## 1) Component Diagram
```mermaid
flowchart LR
    subgraph Java Services
      ORCHC["OrchestrationController"]
      ORCHS["OrchestrationService"]
      INGC["IngestionController"]
      INGS["IngestionJobService"]
      SEC["SecEdgarClient"]
      FILEREPO["FilingRepository"]
    end

    subgraph Python Services
      IDXAPI["Indexing API Routes"]
      IDXSVC["Indexing Service"]
      AIAPI["AI API Routes"]
      LG["LangGraph Workflow"]
      EVAL["EvaluationRunner"]
      RISK["Risk Sanitizer"]
    end

    ORCHC --> ORCHS
    ORCHS --> INGC
    ORCHS --> IDXAPI
    ORCHS --> AIAPI

    INGC --> INGS
    INGS --> SEC
    INGS --> FILEREPO

    IDXAPI --> IDXSVC
    AIAPI --> LG
    AIAPI --> EVAL
    AIAPI --> RISK
```

## 2) Class Diagram
```mermaid
classDiagram
    class FilingEntity {
      accessionNo
      cik
      ticker
      formType
      filingDate
      sourceUrl
      primaryDoc
    }

    class IngestionRunEntity {
      runId
      status
      fetchedCount
      insertedCount
      updatedCount
      skippedCount
      failedCount
      startedAt
      completedAt
    }

    class LangGraphSwarmWorkflow {
      analyze(query, ticker, topK, maxEvidence)
      _filing_agent()
      _sec_facts_agent()
      _news_agent()
      _research_aggregator()
      _critic_evaluate()
      _compare_candidates()
      _write_final()
    }

    class EvaluationRunner {
      run(datasetPath, topK, maxEvidence)
      _aggregate_metrics()
      _write_csv()
      _write_summary()
      _generate_svgs()
    }

    class PythonVectorIndexer {
      search(query, topK, ticker)
      index_batch(filings)
      reset_collection()
      stats()
    }

    IngestionRunEntity --> FilingEntity
    LangGraphSwarmWorkflow --> PythonVectorIndexer
    EvaluationRunner --> LangGraphSwarmWorkflow
```

## 3) Sequence Diagram: Ingestion Request
```mermaid
sequenceDiagram
    participant UI as Streamlit UI
    participant ORCH as Orchestration API
    participant ING as Ingestion API
    participant SEC as SEC EDGAR
    participant DB as Filing Metadata DB

    UI->>ORCH: start ingestion (ticker/cik)
    ORCH->>ING: POST /v1/ingestion/run
    ING->>SEC: fetch filings
    ING->>DB: upsert metadata + run counters
    ING-->>ORCH: runId + status
    ORCH-->>UI: ingestion status
```

## 4) Sequence Diagram: Agentic Analysis
```mermaid
sequenceDiagram
    participant UI as Streamlit UI
    participant API as /v1/agents/analyze
    participant WF as LangGraph Workflow
    participant FIL as Filing Agent
    participant FAC as SEC Facts Agent
    participant NEW as News Agent
    participant CRT as Critic/Comparison
    participant WRT as Writer

    UI->>API: analyze(query, ticker)
    API->>WF: execute workflow
    par specialist pass
      WF->>FIL: retrieve + summarize
      WF->>FAC: retrieve + summarize
      WF->>NEW: retrieve + summarize
    end
    WF->>CRT: evaluate + select
    CRT-->>WF: confidence + strategy
    WF->>WRT: compose final memo
    WRT-->>API: summary + citations + token/cost
    API-->>UI: response + trace
```

## 5) State Diagram: Analysis Lifecycle
```mermaid
stateDiagram-v2
    [*] --> Received
    Received --> Sanitized
    Sanitized --> Retrieval
    Retrieval --> Aggregation
    Aggregation --> CriticReview
    CriticReview --> Comparison
    Comparison --> Writer
    Writer --> Completed
    CriticReview --> Failed: hard failure
    Retrieval --> Failed: source failures
    Failed --> [*]
    Completed --> [*]
```

## 6) UML Implementation Notes

### Package Ownership
- Java ingestion service
  - `controller`, `service`, `client`, `domain`, `repository`, `config`
- Java orchestration service
  - `controller`, `service`, `client`, `domain`, `config`
- Python indexing service
  - `app.main`, `app.routes`, `app.settings`
- Python AI research team service
  - `api`, `agents`, `retrieval`, `risk`, `eval`, `schemas`, `core`
- Streamlit demo UI
  - `services/demo-ui/app.py`

### Runtime Contracts (UML-relevant)
- Ingestion
  - `POST /v1/ingestion/run`
  - `GET /v1/ingestion/runs/{runId}`
  - `GET /v1/filings`
- Indexing
  - `POST /v1/indexing/index-batch`
  - `POST /v1/indexing/reset`
  - `GET /v1/indexing/stats`
- Analysis / Eval
  - `POST /v1/agents/analyze`
  - `POST /v1/eval/run`

### Analysis Object Model
- Research-agent output shape:
  - `count`, `top_items`, `summary_text`, `token_usage`, `estimated_cost_usd`
- Query-level output shape:
  - `summary`, `memo_markdown`, `evidence`, `trace`, `confidence`, `token/cost totals`

### Evaluation Object Model
- Per-sample attributes:
  - `query`, `ticker`, `sanitized_query`, `evidence_count`, `citation_coverage`, `confidence_score`, `latency`
  - `input_tokens`, `output_tokens`, `agent_token_usage_json`, `news_result_count`, `error_message`
- Aggregate summary:
  - retrieval hit rate, citation coverage, confidence, latency, token metrics, quality gate
