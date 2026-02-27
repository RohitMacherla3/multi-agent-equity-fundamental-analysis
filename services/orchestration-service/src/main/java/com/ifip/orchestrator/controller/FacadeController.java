package com.ifip.orchestrator.controller;

import java.util.Map;
import java.util.UUID;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClient;

@RestController
public class FacadeController {

    private final RestClient ingestionClient;
    private final RestClient indexingClient;
    private final RestClient aiClient;

    public FacadeController(
        @Qualifier("ingestionRestClient") RestClient ingestionClient,
        @Qualifier("indexingRestClient") RestClient indexingClient,
        @Qualifier("aiRestClient") RestClient aiClient
    ) {
        this.ingestionClient = ingestionClient;
        this.indexingClient = indexingClient;
        this.aiClient = aiClient;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of(
            "status", "ok",
            "service", "orchestration-service"
        );
    }

    @PostMapping(value = "/v1/ingestion/run", consumes = MediaType.APPLICATION_JSON_VALUE)
    public String runIngestion(@RequestBody String payload) {
        return ingestionClient.post()
            .uri("/v1/ingestion/run")
            .contentType(MediaType.APPLICATION_JSON)
            .body(payload)
            .retrieve()
            .body(String.class);
    }

    @GetMapping("/v1/ingestion/runs/{runId}")
    public String getIngestionRun(@PathVariable UUID runId) {
        return ingestionClient.get()
            .uri("/v1/ingestion/runs/{runId}", runId)
            .retrieve()
            .body(String.class);
    }

    @GetMapping("/v1/filings")
    public String getFilings(@RequestParam(name = "limit", defaultValue = "100") int limit) {
        return ingestionClient.get()
            .uri(uri -> uri.path("/v1/filings").queryParam("limit", limit).build())
            .retrieve()
            .body(String.class);
    }

    @PostMapping(value = "/v1/indexing/reset", consumes = MediaType.APPLICATION_JSON_VALUE)
    public String resetIndex(@RequestBody(required = false) String payload) {
        return indexingClient.post()
            .uri("/v1/indexing/reset")
            .contentType(MediaType.APPLICATION_JSON)
            .body(payload == null ? "{}" : payload)
            .retrieve()
            .body(String.class);
    }

    @PostMapping(value = "/v1/indexing/index-batch", consumes = MediaType.APPLICATION_JSON_VALUE)
    public String indexBatch(@RequestBody String payload) {
        return indexingClient.post()
            .uri("/v1/indexing/index-batch")
            .contentType(MediaType.APPLICATION_JSON)
            .body(payload)
            .retrieve()
            .body(String.class);
    }

    @GetMapping("/v1/indexing/stats")
    public String indexStats() {
        return indexingClient.get()
            .uri("/v1/indexing/stats")
            .retrieve()
            .body(String.class);
    }

    @PostMapping(value = "/v1/agents/analyze", consumes = MediaType.APPLICATION_JSON_VALUE)
    public String analyze(@RequestBody String payload) {
        return aiClient.post()
            .uri("/v1/agents/analyze")
            .contentType(MediaType.APPLICATION_JSON)
            .body(payload)
            .retrieve()
            .body(String.class);
    }

    @PostMapping(value = "/v1/eval/run", consumes = MediaType.APPLICATION_JSON_VALUE)
    public String runEval(@RequestBody String payload) {
        return aiClient.post()
            .uri("/v1/eval/run")
            .contentType(MediaType.APPLICATION_JSON)
            .body(payload)
            .retrieve()
            .body(String.class);
    }
}
