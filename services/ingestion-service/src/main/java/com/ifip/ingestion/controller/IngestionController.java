package com.ifip.ingestion.controller;

import com.ifip.ingestion.config.IngestionProperties;
import com.ifip.ingestion.domain.IngestionFailureEntity;
import com.ifip.ingestion.domain.IngestionRunEntity;
import com.ifip.ingestion.service.IngestionJobService;
import jakarta.validation.Valid;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/v1/ingestion")
public class IngestionController {

    private final IngestionJobService ingestionJobService;
    private final IngestionProperties properties;

    public IngestionController(IngestionJobService ingestionJobService, IngestionProperties properties) {
        this.ingestionJobService = ingestionJobService;
        this.properties = properties;
    }

    @PostMapping("/run")
    public ResponseEntity<Map<String, Object>> run(@Valid @RequestBody IngestionRunRequest request) {
        int maxPerCik = request.maxPerCik() == null ? properties.getDefaultMaxPerCik() : request.maxPerCik();
        boolean includeDocuments = request.includeDocuments() == null
            ? properties.isDefaultIncludeDocuments()
            : request.includeDocuments();

        UUID runId = ingestionJobService.runIncrementalIngestion(request.ciks(), maxPerCik, includeDocuments);
        return ResponseEntity.accepted().body(Map.of("runId", runId));
    }

    @GetMapping("/runs/{runId}")
    public IngestionRunResponse getRun(@PathVariable UUID runId) {
        IngestionRunEntity run = ingestionJobService.getRun(runId)
            .orElseThrow(() -> new IllegalArgumentException("Run not found: " + runId));

        List<IngestionRunResponse.FailureItem> failures = ingestionJobService.getRunFailures(runId)
            .stream()
            .map(this::toFailureItem)
            .toList();

        return new IngestionRunResponse(
            run.getRunId(),
            run.getStatus(),
            run.getStartedAt(),
            run.getCompletedAt(),
            run.getFetchedCount(),
            run.getInsertedCount(),
            run.getUpdatedCount(),
            run.getSkippedCount(),
            run.getFailedCount(),
            run.getErrorSummary(),
            failures
        );
    }

    private IngestionRunResponse.FailureItem toFailureItem(IngestionFailureEntity entity) {
        return new IngestionRunResponse.FailureItem(
            entity.getAccessionNo(),
            entity.getFailureCode(),
            entity.getFailureReason(),
            entity.getCreatedAt()
        );
    }
}
