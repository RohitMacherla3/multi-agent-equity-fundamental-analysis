# Architecture Diagrams: Multi-Agent Equity Fundamental Analysis

This document is the one-stop reference for runtime architecture, service boundaries, data flow, and agentic workflow behavior.

## 1) System Context
```mermaid
flowchart LR
    U["Analyst"] --> UI["Streamlit Demo UI"]
    UI --> ORCH["orchestration-service (Java facade)"]

    ORCH --> ING["ingestion-service (Java)"]
    ORCH --> IDX["indexing-service (Python)"]
    ORCH --> AI["ai-research-team-service (Python)"]

    ING --> SEC["SEC EDGAR APIs"]
    ING --> META[("Filing Metadata DB")]
    ING --> RAW[("Raw Filing Store")]

    IDX --> META
    IDX --> RAW
    IDX --> CHROMA[("Chroma Vector DB")]
    IDX --> EMB["OpenAI Embeddings"]

    AI --> CHROMA
    AI --> LLM["OpenAI LLM"]
    AI --> FACTS["SEC Company Facts API"]
    AI --> NEWS["Tavily Search API"]
    AI --> EVAL[("Evaluation Artifacts CSV/JSON/SVG")]
```

## 2) Service Boundary View
```mermaid
flowchart TB
    subgraph Java
      ORCH["orchestration-service"]
      ING["ingestion-service"]
    end

    subgraph Python
      IDX["indexing-service"]
      AI["ai-research-team-service"]
      UI["demo-ui"]
    end

    ORCH --> ING
    ORCH --> IDX
    ORCH --> AI
    UI --> ORCH
```

## 3) Ingestion + Indexing Pipeline
```mermaid
sequenceDiagram
    autonumber
    participant User as Analyst
    participant UI as Streamlit
    participant ORCH as Orchestration Service
    participant ING as Ingestion Service
    participant SEC as SEC EDGAR
    participant META as Filing Metadata DB
    participant RAW as Raw Filing Store
    participant IDX as Indexing Service
    participant CHROMA as Chroma DB

    User->>UI: Select ticker + filings count
    UI->>ORCH: Trigger process
    ORCH->>ING: POST /v1/ingestion/run
    ING->>SEC: Fetch submissions/filings
    ING->>META: Upsert filing metadata
    ING->>RAW: Store raw filing payload
    ORCH->>IDX: POST /v1/indexing/index-batch
    IDX->>META: Read filing metadata
    IDX->>RAW: Read filing content
    IDX->>CHROMA: Write embedded chunks
    ORCH-->>UI: Process complete
```

## 4) Agentic Analysis Architecture
```mermaid
flowchart LR
    Q["Query + Ticker"] --> FIL["Filing Agent"]
    Q --> FAC["SEC Facts Agent"]
    Q --> NEW["News Agent"]

    FIL --> AGG["Research Aggregator"]
    FAC --> AGG
    NEW --> AGG

    AGG --> CRT["Critic Agent"]
    CRT --> CMP["Comparison Agent"]
    CMP --> WRT["Writer Agent"]

    FIL --> TRACE[("Agent Trace")]
    FAC --> TRACE
    NEW --> TRACE
    CRT --> TRACE
    WRT --> TRACE
```

## 5) Analysis Request Sequence
```mermaid
sequenceDiagram
    autonumber
    participant UI as Streamlit
    participant AI as AI Research Team
    participant CHROMA as Chroma
    participant FACTS as SEC Facts API
    participant NEWS as Tavily
    participant LLM as OpenAI LLM

    UI->>AI: POST /v1/agents/analyze
    par Specialist Retrieval
      AI->>CHROMA: Filing evidence retrieval
      AI->>FACTS: Company facts retrieval
      AI->>NEWS: News retrieval
    end
    AI->>LLM: Specialist summaries
    AI->>LLM: Aggregation + critic + comparison
    AI->>LLM: Final writer memo
    AI-->>UI: Summary + evidence + confidence + trace + tokens/cost
```

## 6) Evaluation and Reporting Flow
```mermaid
flowchart LR
    D["Eval Dataset (ticker x prompt)"] --> R["Evaluation Runner"]
    R --> C["Per-sample CSV"]
    R --> S["Summary JSON"]
    R --> V["SVG Dashboard Pack"]

    C --> M1["retrieval/citation/confidence/latency"]
    C --> M2["input/output tokens + agent token usage"]
    S --> G["quality gate checks"]
```

## 7) Data Reuse Decision
```mermaid
flowchart TD
    A["Ticker selected"] --> B{"Existing filings already sufficient?"}
    B -- Yes --> C["Reuse existing filings"]
    B -- No --> D["Run ingestion"]
    D --> E["Update metadata/raw store"]
    C --> F["Index and analyze"]
    E --> F
```

## 8) Storage Surfaces
- Filing metadata: structured filing/run records used by ingestion and indexing.
- Raw filing store: source payload for chunking/indexing.
- Chroma vector DB: semantic retrieval surface for filing evidence.
- Eval artifacts: dataset/eval CSV/JSON + SVG charts in `data/eval-reports`.
