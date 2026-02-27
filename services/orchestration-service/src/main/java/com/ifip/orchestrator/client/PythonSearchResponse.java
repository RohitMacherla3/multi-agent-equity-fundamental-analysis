package com.ifip.orchestrator.client;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public record PythonSearchResponse(
    String query,
    @JsonProperty("top_k") int topK,
    List<PythonSearchResult> results
) {
}
