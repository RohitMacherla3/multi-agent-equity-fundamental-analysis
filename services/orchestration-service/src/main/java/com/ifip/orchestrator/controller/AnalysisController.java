package com.ifip.orchestrator.controller;

import com.ifip.orchestrator.domain.AnalysisRunRequest;
import com.ifip.orchestrator.domain.AnalysisRunStatus;
import com.ifip.orchestrator.domain.AnalysisRunView;
import com.ifip.orchestrator.domain.AnalysisStartResponse;
import com.ifip.orchestrator.service.AnalysisOrchestratorService;
import jakarta.validation.Valid;
import java.util.UUID;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/v1/analysis")
public class AnalysisController {

    private final AnalysisOrchestratorService orchestratorService;

    public AnalysisController(AnalysisOrchestratorService orchestratorService) {
        this.orchestratorService = orchestratorService;
    }

    @PostMapping("/run")
    public ResponseEntity<AnalysisStartResponse> run(@Valid @RequestBody AnalysisRunRequest request) {
        UUID runId = orchestratorService.startRun(request);
        return ResponseEntity.accepted().body(new AnalysisStartResponse(runId, AnalysisRunStatus.QUEUED));
    }

    @GetMapping("/runs/{runId}")
    public AnalysisRunView getRun(@PathVariable UUID runId) {
        return orchestratorService.getRun(runId)
            .orElseThrow(() -> new IllegalArgumentException("Analysis run not found: " + runId));
    }
}
