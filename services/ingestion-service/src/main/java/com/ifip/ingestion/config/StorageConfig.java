package com.ifip.ingestion.config;

import com.ifip.ingestion.client.LocalRawDocumentStorage;
import com.ifip.ingestion.client.RawDocumentStorage;
import java.nio.file.Path;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class StorageConfig {

    @Bean
    RawDocumentStorage rawDocumentStorage(IngestionProperties properties) {
        return new LocalRawDocumentStorage(Path.of(properties.getRawStoragePath()));
    }
}
