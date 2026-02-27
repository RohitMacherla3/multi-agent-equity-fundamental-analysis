package com.ifip.ingestion.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
import jakarta.persistence.Table;
import java.time.Instant;
import java.time.LocalDate;

@Entity
@Table(name = "filings")
public class FilingEntity {

    @Id
    @Column(name = "accession_no", nullable = false, updatable = false)
    private String accessionNo;

    @Column(name = "cik", nullable = false)
    private String cik;

    @Column(name = "ticker")
    private String ticker;

    @Column(name = "company_name")
    private String companyName;

    @Column(name = "form_type", nullable = false)
    private String formType;

    @Column(name = "filing_date", nullable = false)
    private LocalDate filingDate;

    @Column(name = "primary_doc")
    private String primaryDoc;

    @Column(name = "source_url")
    private String sourceUrl;

    @Column(name = "checksum")
    private String checksum;

    @Column(name = "storage_uri")
    private String storageUri;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    public static FilingEntity fromRecord(FilingRecord record) {
        FilingEntity entity = new FilingEntity();
        entity.accessionNo = record.accessionNo();
        entity.cik = record.cik();
        entity.ticker = record.ticker();
        entity.companyName = record.companyName();
        entity.formType = record.formType();
        entity.filingDate = record.filingDate();
        entity.primaryDoc = record.primaryDoc();
        entity.sourceUrl = record.sourceUrl();
        return entity;
    }

    public void updateFrom(FilingRecord record) {
        this.cik = record.cik();
        this.ticker = record.ticker();
        this.companyName = record.companyName();
        this.formType = record.formType();
        this.filingDate = record.filingDate();
        this.primaryDoc = record.primaryDoc();
        this.sourceUrl = record.sourceUrl();
    }

    @PrePersist
    void onCreate() {
        Instant now = Instant.now();
        this.createdAt = now;
        this.updatedAt = now;
    }

    @PreUpdate
    void onUpdate() {
        this.updatedAt = Instant.now();
    }

    public String getAccessionNo() {
        return accessionNo;
    }

    public String getCik() {
        return cik;
    }

    public String getTicker() {
        return ticker;
    }

    public String getCompanyName() {
        return companyName;
    }

    public String getFormType() {
        return formType;
    }

    public LocalDate getFilingDate() {
        return filingDate;
    }

    public String getPrimaryDoc() {
        return primaryDoc;
    }

    public String getSourceUrl() {
        return sourceUrl;
    }

    public String getChecksum() {
        return checksum;
    }

    public void setChecksum(String checksum) {
        this.checksum = checksum;
    }

    public String getStorageUri() {
        return storageUri;
    }

    public void setStorageUri(String storageUri) {
        this.storageUri = storageUri;
    }
}
