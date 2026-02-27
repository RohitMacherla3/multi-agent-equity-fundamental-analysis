package com.ifip.ingestion.controller;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotEmpty;
import java.util.List;

public record IngestionRunRequest(
    @NotEmpty(message = "ciks must not be empty")
    List<String> ciks,

    @Min(1)
    @Max(200)
    Integer maxPerCik,

    Boolean includeDocuments
) {
}
