package com.ifip.ingestion.batch;

import com.ifip.ingestion.config.IngestionProperties;
import com.ifip.ingestion.service.IngestionJobService;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class IngestionScheduler {

    private static final Logger LOGGER = LoggerFactory.getLogger(IngestionScheduler.class);

    private final IngestionProperties properties;
    private final IngestionJobService ingestionJobService;

    public IngestionScheduler(IngestionProperties properties, IngestionJobService ingestionJobService) {
        this.properties = properties;
        this.ingestionJobService = ingestionJobService;
    }

    @Scheduled(fixedDelayString = "${ingestion.scheduler-fixed-delay-ms:3600000}")
    public void runScheduledIngestion() {
        if (!properties.isSchedulerEnabled()) {
            return;
        }
        List<String> ciks = properties.getDefaultCiks();
        if (ciks == null || ciks.isEmpty()) {
            LOGGER.warn("Scheduler enabled but no default CIKs configured");
            return;
        }
        LOGGER.info("Running scheduled ingestion for {} CIKs", ciks.size());
        ingestionJobService.runIncrementalIngestion(
            ciks,
            properties.getDefaultMaxPerCik(),
            properties.isDefaultIncludeDocuments()
        );
    }
}
