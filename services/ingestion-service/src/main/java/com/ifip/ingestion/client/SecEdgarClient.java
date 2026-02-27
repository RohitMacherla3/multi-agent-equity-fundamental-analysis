package com.ifip.ingestion.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.ifip.ingestion.domain.FilingRecord;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.reactive.function.client.WebClient;

@Component
public class SecEdgarClient {

    private final WebClient secDataWebClient;
    private final WebClient secArchiveWebClient;

    public SecEdgarClient(
        @Qualifier("secDataWebClient") WebClient secDataWebClient,
        @Qualifier("secArchiveWebClient") WebClient secArchiveWebClient
    ) {
        this.secDataWebClient = secDataWebClient;
        this.secArchiveWebClient = secArchiveWebClient;
    }

    public List<FilingRecord> fetchRecentFilings(String cik, String ticker, int maxPerCik) {
        String paddedCik = String.format("%010d", Long.parseLong(cik));

        JsonNode root = secDataWebClient.get()
            .uri(uriBuilder -> uriBuilder.path("/submissions/CIK" + paddedCik + ".json").build())
            .retrieve()
            .bodyToMono(JsonNode.class)
            .block();

        if (root == null) {
            return List.of();
        }

        String companyName = text(root.get("name"));
        JsonNode recent = root.path("filings").path("recent");

        JsonNode accessionNumbers = recent.path("accessionNumber");
        JsonNode forms = recent.path("form");
        JsonNode filingDates = recent.path("filingDate");
        JsonNode primaryDocuments = recent.path("primaryDocument");

        int size = Math.min(accessionNumbers.size(), maxPerCik);
        List<FilingRecord> records = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            String accessionNo = text(accessionNumbers.get(i));
            String formType = text(forms.get(i));
            String filingDateRaw = text(filingDates.get(i));
            String primaryDoc = text(primaryDocuments.get(i));

            if (accessionNo.isBlank() || filingDateRaw.isBlank()) {
                continue;
            }

            String sourceUrl = resolvePrimaryDocumentUrl(cik, accessionNo, primaryDoc);
            records.add(new FilingRecord(
                accessionNo,
                cik,
                ticker,
                companyName,
                formType,
                LocalDate.parse(filingDateRaw),
                primaryDoc,
                sourceUrl
            ));
        }
        return records;
    }

    public RawDocument downloadDocument(FilingRecord record) {
        byte[] bytes = secArchiveWebClient.get()
            .uri(record.sourceUrl())
            .retrieve()
            .bodyToMono(byte[].class)
            .block();

        if (bytes == null) {
            throw new IllegalStateException("Downloaded document payload is empty for " + record.accessionNo());
        }

        return new RawDocument(
            record.primaryDoc() == null || record.primaryDoc().isBlank() ? record.accessionNo() + ".txt" : record.primaryDoc(),
            bytes,
            sha256(bytes)
        );
    }

    private String resolvePrimaryDocumentUrl(String cik, String accessionNo, String primaryDoc) {
        String cikNoLeadingZero = stripLeadingZeros(cik);
        String accessionWithoutDashes = accessionNo.replace("-", "");
        String doc = (primaryDoc == null || primaryDoc.isBlank()) ? accessionNo + ".txt" : primaryDoc;
        return "https://www.sec.gov/Archives/edgar/data/"
            + cikNoLeadingZero + "/" + accessionWithoutDashes + "/" + doc;
    }

    private String stripLeadingZeros(String value) {
        String stripped = value.replaceFirst("^0+(?!$)", "");
        return stripped.isBlank() ? "0" : stripped;
    }

    private String text(JsonNode node) {
        return node == null || node.isNull() ? "" : node.asText("").trim();
    }

    private String sha256(byte[] payload) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(payload);
            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }
}
