package com.ifip.orchestrator.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "orchestrator")
public class OrchestratorProperties {

    private String ingestionBaseUrl = "http://localhost:8080";
    private String indexingBaseUrl = "http://localhost:8002";
    private String aiBaseUrl = "http://localhost:8000";
    private int defaultTopK = 5;

    public String getIngestionBaseUrl() {
        return ingestionBaseUrl;
    }

    public void setIngestionBaseUrl(String ingestionBaseUrl) {
        this.ingestionBaseUrl = ingestionBaseUrl;
    }

    public String getIndexingBaseUrl() {
        return indexingBaseUrl;
    }

    public void setIndexingBaseUrl(String indexingBaseUrl) {
        this.indexingBaseUrl = indexingBaseUrl;
    }

    public String getAiBaseUrl() {
        return aiBaseUrl;
    }

    public void setAiBaseUrl(String aiBaseUrl) {
        this.aiBaseUrl = aiBaseUrl;
    }

    public int getDefaultTopK() {
        return defaultTopK;
    }

    public void setDefaultTopK(int defaultTopK) {
        this.defaultTopK = defaultTopK;
    }
}
