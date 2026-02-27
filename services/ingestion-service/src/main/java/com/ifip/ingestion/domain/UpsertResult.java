package com.ifip.ingestion.domain;

public enum UpsertResult {
    INSERTED,
    UPDATED,
    SKIPPED;

    public boolean isInsertOrUpdate() {
        return this == INSERTED || this == UPDATED;
    }
}
