package com.ifip.orchestrator.domain;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

public record AnalysisRunView(
    UUID runId,
    AnalysisRunStatus status,
    String query,
    String ticker,
    int topK,
    Instant startedAt,
    Instant completedAt,
    String error,
    List<AnalysisItem> results
) {
}
