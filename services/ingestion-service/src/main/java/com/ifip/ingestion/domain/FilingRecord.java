package com.ifip.ingestion.domain;

import java.time.LocalDate;

public record FilingRecord(
    String accessionNo,
    String cik,
    String ticker,
    String companyName,
    String formType,
    LocalDate filingDate,
    String primaryDoc,
    String sourceUrl
) {
}
