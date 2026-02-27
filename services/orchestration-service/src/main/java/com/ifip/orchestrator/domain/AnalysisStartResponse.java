package com.ifip.orchestrator.domain;

import java.util.UUID;

public record AnalysisStartResponse(UUID runId, AnalysisRunStatus status) {
}
