package com.ifip.ingestion.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "ingestion_failures")
public class IngestionFailureEntity {

    @Id
    @Column(name = "id", nullable = false, updatable = false)
    private UUID id;

    @Column(name = "run_id", nullable = false)
    private UUID runId;

    @Column(name = "accession_no")
    private String accessionNo;

    @Column(name = "failure_code", nullable = false)
    private String failureCode;

    @Column(name = "failure_reason", nullable = false)
    private String failureReason;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    public static IngestionFailureEntity of(UUID runId, String accessionNo, String code, String reason) {
        IngestionFailureEntity entity = new IngestionFailureEntity();
        entity.id = UUID.randomUUID();
        entity.runId = runId;
        entity.accessionNo = accessionNo;
        entity.failureCode = code;
        entity.failureReason = reason;
        entity.createdAt = Instant.now();
        return entity;
    }

    public UUID getId() {
        return id;
    }

    public UUID getRunId() {
        return runId;
    }

    public String getAccessionNo() {
        return accessionNo;
    }

    public String getFailureCode() {
        return failureCode;
    }

    public String getFailureReason() {
        return failureReason;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }
}
