package com.ifip.ingestion.controller;

import com.ifip.ingestion.domain.FilingEntity;
import java.time.LocalDate;

public record FilingResponse(
    String accessionNo,
    String cik,
    String ticker,
    String companyName,
    String formType,
    LocalDate filingDate,
    String primaryDoc,
    String sourceUrl,
    String checksum,
    String storageUri
) {
    public static FilingResponse from(FilingEntity entity) {
        return new FilingResponse(
            entity.getAccessionNo(),
            entity.getCik(),
            entity.getTicker(),
            entity.getCompanyName(),
            entity.getFormType(),
            entity.getFilingDate(),
            entity.getPrimaryDoc(),
            entity.getSourceUrl(),
            entity.getChecksum(),
            entity.getStorageUri()
        );
    }
}
