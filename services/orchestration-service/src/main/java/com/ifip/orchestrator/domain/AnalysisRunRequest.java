package com.ifip.orchestrator.domain;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;

public record AnalysisRunRequest(
    @NotBlank(message = "query must not be blank")
    String query,

    String ticker,

    @Min(value = 1, message = "topK must be >= 1")
    @Max(value = 50, message = "topK must be <= 50")
    Integer topK
) {
}
