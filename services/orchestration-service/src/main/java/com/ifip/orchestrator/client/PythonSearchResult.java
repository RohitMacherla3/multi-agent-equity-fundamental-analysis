package com.ifip.orchestrator.client;

import com.fasterxml.jackson.annotation.JsonProperty;

public record PythonSearchResult(
    @JsonProperty("accession_no") String accessionNo,
    String ticker,
    @JsonProperty("form_type") String formType,
    @JsonProperty("section_name") String sectionName,
    @JsonProperty("chunk_index") int chunkIndex,
    @JsonProperty("text_preview") String textPreview,
    double score
) {
}
