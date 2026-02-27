package com.ifip.orchestrator.client;

import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.client.RestClient;
import org.springframework.web.util.UriBuilder;

@Component
public class PythonIndexingClient {

    private final RestClient restClient;

    public PythonIndexingClient(@Qualifier("indexingRestClient") RestClient pythonRestClient) {
        this.restClient = pythonRestClient;
    }

    public PythonSearchResponse search(String query, int topK, String ticker) {
        return restClient.get()
            .uri(uriBuilder -> {
                UriBuilder builder = uriBuilder
                    .path("/v1/indexing/search")
                    .queryParam("q", query)
                    .queryParam("topK", topK);
                if (ticker != null && !ticker.isBlank()) {
                    builder = builder.queryParam("ticker", ticker);
                }
                return builder.build();
            })
            .retrieve()
            .body(PythonSearchResponse.class);
    }
}
