package com.ifip.orchestrator.service;

import com.ifip.orchestrator.client.PythonIndexingClient;
import com.ifip.orchestrator.client.PythonSearchResponse;
import com.ifip.orchestrator.domain.AnalysisItem;
import com.ifip.orchestrator.domain.AnalysisRunRequest;
import com.ifip.orchestrator.domain.AnalysisRunStatus;
import com.ifip.orchestrator.domain.AnalysisRunView;
import com.ifip.orchestrator.config.OrchestratorProperties;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

@Service
public class AnalysisOrchestratorService {

    private final PythonIndexingClient pythonIndexingClient;
    private final OrchestratorProperties properties;
    private final ConcurrentHashMap<UUID, MutableRun> runs = new ConcurrentHashMap<>();

    public AnalysisOrchestratorService(PythonIndexingClient pythonIndexingClient, OrchestratorProperties properties) {
        this.pythonIndexingClient = pythonIndexingClient;
        this.properties = properties;
    }

    public UUID startRun(AnalysisRunRequest request) {
        UUID runId = UUID.randomUUID();
        int topK = request.topK() == null ? properties.getDefaultTopK() : request.topK();

        MutableRun run = new MutableRun(
            runId,
            AnalysisRunStatus.QUEUED,
            request.query(),
            request.ticker(),
            topK,
            Instant.now(),
            null,
            null,
            List.of()
        );
        runs.put(runId, run);

        executeAsync(runId);
        return runId;
    }

    public Optional<AnalysisRunView> getRun(UUID runId) {
        return Optional.ofNullable(runs.get(runId)).map(MutableRun::toView);
    }

    @Async
    protected CompletableFuture<Void> executeAsync(UUID runId) {
        MutableRun run = runs.get(runId);
        if (run == null) {
            return CompletableFuture.completedFuture(null);
        }

        run.status = AnalysisRunStatus.RUNNING;
        try {
            PythonSearchResponse response = pythonIndexingClient.search(run.query, run.topK, run.ticker);
            run.results = response.results().stream()
                .map(r -> new AnalysisItem(
                    r.accessionNo(),
                    r.ticker(),
                    r.formType(),
                    r.sectionName(),
                    r.chunkIndex(),
                    r.textPreview(),
                    r.score()
                ))
                .toList();
            run.status = AnalysisRunStatus.SUCCEEDED;
        } catch (Exception ex) {
            run.status = AnalysisRunStatus.FAILED;
            run.error = ex.getMessage();
        } finally {
            run.completedAt = Instant.now();
        }
        return CompletableFuture.completedFuture(null);
    }

    private static class MutableRun {
        private final UUID runId;
        private AnalysisRunStatus status;
        private final String query;
        private final String ticker;
        private final int topK;
        private final Instant startedAt;
        private Instant completedAt;
        private String error;
        private List<AnalysisItem> results;

        private MutableRun(
            UUID runId,
            AnalysisRunStatus status,
            String query,
            String ticker,
            int topK,
            Instant startedAt,
            Instant completedAt,
            String error,
            List<AnalysisItem> results
        ) {
            this.runId = runId;
            this.status = status;
            this.query = query;
            this.ticker = ticker;
            this.topK = topK;
            this.startedAt = startedAt;
            this.completedAt = completedAt;
            this.error = error;
            this.results = results;
        }

        private AnalysisRunView toView() {
            return new AnalysisRunView(runId, status, query, ticker, topK, startedAt, completedAt, error, results);
        }
    }
}
