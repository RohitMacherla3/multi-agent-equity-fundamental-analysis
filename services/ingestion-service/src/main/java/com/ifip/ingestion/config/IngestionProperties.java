package com.ifip.ingestion.config;

import java.util.ArrayList;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "ingestion")
public class IngestionProperties {

    private String userAgent = "IFIPResearchBot/1.0 (rohitmacherla125@gmail.com)";
    private boolean schedulerEnabled = false;
    private long schedulerFixedDelayMs = 3_600_000;
    private List<String> defaultCiks = new ArrayList<>();
    private int defaultMaxPerCik = 25;
    private boolean defaultIncludeDocuments = true;
    private String rawStoragePath = "data/raw-filings";
    private int secDataMaxInMemoryMb = 16;
    private int secArchiveMaxInMemoryMb = 25;

    public String getUserAgent() {
        return userAgent;
    }

    public void setUserAgent(String userAgent) {
        this.userAgent = userAgent;
    }

    public boolean isSchedulerEnabled() {
        return schedulerEnabled;
    }

    public void setSchedulerEnabled(boolean schedulerEnabled) {
        this.schedulerEnabled = schedulerEnabled;
    }

    public long getSchedulerFixedDelayMs() {
        return schedulerFixedDelayMs;
    }

    public void setSchedulerFixedDelayMs(long schedulerFixedDelayMs) {
        this.schedulerFixedDelayMs = schedulerFixedDelayMs;
    }

    public List<String> getDefaultCiks() {
        return defaultCiks;
    }

    public void setDefaultCiks(List<String> defaultCiks) {
        this.defaultCiks = defaultCiks;
    }

    public int getDefaultMaxPerCik() {
        return defaultMaxPerCik;
    }

    public void setDefaultMaxPerCik(int defaultMaxPerCik) {
        this.defaultMaxPerCik = defaultMaxPerCik;
    }

    public boolean isDefaultIncludeDocuments() {
        return defaultIncludeDocuments;
    }

    public void setDefaultIncludeDocuments(boolean defaultIncludeDocuments) {
        this.defaultIncludeDocuments = defaultIncludeDocuments;
    }

    public String getRawStoragePath() {
        return rawStoragePath;
    }

    public void setRawStoragePath(String rawStoragePath) {
        this.rawStoragePath = rawStoragePath;
    }

    public int getSecDataMaxInMemoryMb() {
        return secDataMaxInMemoryMb;
    }

    public void setSecDataMaxInMemoryMb(int secDataMaxInMemoryMb) {
        this.secDataMaxInMemoryMb = secDataMaxInMemoryMb;
    }

    public int getSecArchiveMaxInMemoryMb() {
        return secArchiveMaxInMemoryMb;
    }

    public void setSecArchiveMaxInMemoryMb(int secArchiveMaxInMemoryMb) {
        this.secArchiveMaxInMemoryMb = secArchiveMaxInMemoryMb;
    }
}
