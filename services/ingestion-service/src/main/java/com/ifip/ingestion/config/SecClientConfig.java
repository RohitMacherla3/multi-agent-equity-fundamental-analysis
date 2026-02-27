package com.ifip.ingestion.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class SecClientConfig {

    @Bean
    @Qualifier("secDataWebClient")
    WebClient secDataWebClient(IngestionProperties properties) {
        int maxBytes = Math.max(2, properties.getSecDataMaxInMemoryMb()) * 1024 * 1024;
        ExchangeStrategies strategies = ExchangeStrategies.builder()
            .codecs(configurer -> configurer.defaultCodecs().maxInMemorySize(maxBytes))
            .build();
        return WebClient.builder()
            .baseUrl("https://data.sec.gov")
            .defaultHeader("User-Agent", properties.getUserAgent())
            .defaultHeader("Accept", "application/json")
            .exchangeStrategies(strategies)
            .build();
    }

    @Bean
    @Qualifier("secArchiveWebClient")
    WebClient secArchiveWebClient(IngestionProperties properties) {
        int maxBytes = Math.max(8, properties.getSecArchiveMaxInMemoryMb()) * 1024 * 1024;
        ExchangeStrategies strategies = ExchangeStrategies.builder()
            .codecs(configurer -> configurer.defaultCodecs().maxInMemorySize(maxBytes))
            .build();
        return WebClient.builder()
            .baseUrl("https://www.sec.gov")
            .defaultHeader("User-Agent", properties.getUserAgent())
            .defaultHeader("Accept", "*/*")
            .exchangeStrategies(strategies)
            .build();
    }
}
