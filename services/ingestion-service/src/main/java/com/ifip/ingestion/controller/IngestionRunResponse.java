package com.ifip.ingestion.controller;

import com.ifip.ingestion.domain.RunStatus;
import java.time.Instant;
import java.util.List;
import java.util.UUID;

public record IngestionRunResponse(
    UUID runId,
    RunStatus status,
    Instant startedAt,
    Instant completedAt,
    int fetchedCount,
    int insertedCount,
    int updatedCount,
    int skippedCount,
    int failedCount,
    String errorSummary,
    List<FailureItem> recentFailures
) {
    public record FailureItem(String accessionNo, String code, String reason, Instant createdAt) {
    }
}
