CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id UUID PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status VARCHAR(32) NOT NULL,
    fetched_count INT NOT NULL DEFAULT 0,
    inserted_count INT NOT NULL DEFAULT 0,
    updated_count INT NOT NULL DEFAULT 0,
    skipped_count INT NOT NULL DEFAULT 0,
    failed_count INT NOT NULL DEFAULT 0,
    error_summary VARCHAR(500)
);

CREATE TABLE IF NOT EXISTS filings (
    accession_no VARCHAR(32) PRIMARY KEY,
    cik VARCHAR(16) NOT NULL,
    ticker VARCHAR(16),
    company_name VARCHAR(255),
    form_type VARCHAR(16) NOT NULL,
    filing_date DATE NOT NULL,
    primary_doc VARCHAR(255),
    source_url TEXT,
    checksum VARCHAR(128),
    storage_uri TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings (ticker);
CREATE INDEX IF NOT EXISTS idx_filings_form_type ON filings (form_type);
CREATE INDEX IF NOT EXISTS idx_filings_filing_date ON filings (filing_date DESC);

CREATE TABLE IF NOT EXISTS ingestion_failures (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL,
    accession_no VARCHAR(32),
    failure_code VARCHAR(64) NOT NULL,
    failure_reason VARCHAR(500) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT fk_ingestion_failures_run FOREIGN KEY (run_id) REFERENCES ingestion_runs (run_id)
);

CREATE INDEX IF NOT EXISTS idx_ingestion_failures_run ON ingestion_failures (run_id, created_at DESC);
