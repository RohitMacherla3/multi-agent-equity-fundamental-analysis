package com.ifip.orchestrator.domain;

public record AnalysisItem(
    String accessionNo,
    String ticker,
    String formType,
    String sectionName,
    int chunkIndex,
    String textPreview,
    double score
) {
}
