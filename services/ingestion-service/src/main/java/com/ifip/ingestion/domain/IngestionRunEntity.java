package com.ifip.ingestion.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "ingestion_runs")
public class IngestionRunEntity {

    @Id
    @Column(name = "run_id", nullable = false, updatable = false)
    private UUID runId;

    @Column(name = "started_at", nullable = false)
    private Instant startedAt;

    @Column(name = "completed_at")
    private Instant completedAt;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false)
    private RunStatus status;

    @Column(name = "fetched_count", nullable = false)
    private int fetchedCount;

    @Column(name = "inserted_count", nullable = false)
    private int insertedCount;

    @Column(name = "updated_count", nullable = false)
    private int updatedCount;

    @Column(name = "skipped_count", nullable = false)
    private int skippedCount;

    @Column(name = "failed_count", nullable = false)
    private int failedCount;

    @Column(name = "error_summary")
    private String errorSummary;

    public static IngestionRunEntity startNew() {
        IngestionRunEntity run = new IngestionRunEntity();
        run.runId = UUID.randomUUID();
        run.startedAt = Instant.now();
        run.status = RunStatus.RUNNING;
        return run;
    }

    public void incrementFetched() {
        this.fetchedCount++;
    }

    public void incrementInserted() {
        this.insertedCount++;
    }

    public void incrementUpdated() {
        this.updatedCount++;
    }

    public void incrementSkipped() {
        this.skippedCount++;
    }

    public void incrementFailed() {
        this.failedCount++;
    }

    public void complete() {
        this.completedAt = Instant.now();
        this.status = this.failedCount > 0 ? RunStatus.PARTIAL_SUCCESS : RunStatus.SUCCEEDED;
    }

    public void fail(String errorSummary) {
        this.completedAt = Instant.now();
        this.status = RunStatus.FAILED;
        this.errorSummary = errorSummary;
    }

    public UUID getRunId() {
        return runId;
    }

    public Instant getStartedAt() {
        return startedAt;
    }

    public Instant getCompletedAt() {
        return completedAt;
    }

    public RunStatus getStatus() {
        return status;
    }

    public int getFetchedCount() {
        return fetchedCount;
    }

    public int getInsertedCount() {
        return insertedCount;
    }

    public int getUpdatedCount() {
        return updatedCount;
    }

    public int getSkippedCount() {
        return skippedCount;
    }

    public int getFailedCount() {
        return failedCount;
    }

    public String getErrorSummary() {
        return errorSummary;
    }
}
