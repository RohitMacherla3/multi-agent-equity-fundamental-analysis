package com.ifip.orchestrator.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

@Configuration
public class ClientConfig {

    @Bean
    RestClient ingestionRestClient(OrchestratorProperties properties) {
        return RestClient.builder()
            .baseUrl(properties.getIngestionBaseUrl())
            .build();
    }

    @Bean
    RestClient indexingRestClient(OrchestratorProperties properties) {
        return RestClient.builder()
            .baseUrl(properties.getIndexingBaseUrl())
            .build();
    }

    @Bean
    RestClient aiRestClient(OrchestratorProperties properties) {
        return RestClient.builder()
            .baseUrl(properties.getAiBaseUrl())
            .build();
    }
}
