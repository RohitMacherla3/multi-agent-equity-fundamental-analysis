package com.ifip.ingestion.service;

import com.ifip.ingestion.client.RawDocument;
import com.ifip.ingestion.client.RawDocumentStorage;
import com.ifip.ingestion.client.SecEdgarClient;
import com.ifip.ingestion.domain.FilingEntity;
import com.ifip.ingestion.domain.FilingRecord;
import com.ifip.ingestion.domain.IngestionFailureEntity;
import com.ifip.ingestion.domain.IngestionRunEntity;
import com.ifip.ingestion.domain.UpsertResult;
import com.ifip.ingestion.repository.FilingRepository;
import com.ifip.ingestion.repository.IngestionFailureRepository;
import com.ifip.ingestion.repository.IngestionRunRepository;
import jakarta.transaction.Transactional;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class IngestionJobService {

    private static final Set<String> SUPPORTED_FORMS = Set.of("10-K", "10-Q", "8-K", "4");

    private static final Map<String, String> TICKER_BY_CIK = Map.of(
        "0000886982", "GS",
        "0000789019", "MSFT",
        "0001045810", "NVDA",
        "0000320193", "AAPL"
    );

    private final SecEdgarClient secEdgarClient;
    private final FilingRepository filingRepository;
    private final IngestionRunRepository ingestionRunRepository;
    private final IngestionFailureRepository ingestionFailureRepository;
    private final RawDocumentStorage rawDocumentStorage;

    public IngestionJobService(
        SecEdgarClient secEdgarClient,
        FilingRepository filingRepository,
        IngestionRunRepository ingestionRunRepository,
        IngestionFailureRepository ingestionFailureRepository,
        RawDocumentStorage rawDocumentStorage
    ) {
        this.secEdgarClient = secEdgarClient;
        this.filingRepository = filingRepository;
        this.ingestionRunRepository = ingestionRunRepository;
        this.ingestionFailureRepository = ingestionFailureRepository;
        this.rawDocumentStorage = rawDocumentStorage;
    }

    @Transactional
    public UUID runIncrementalIngestion(List<String> ciks, int maxPerCik, boolean includeDocuments) {
        IngestionRunEntity run = ingestionRunRepository.save(IngestionRunEntity.startNew());
        try {
            for (String cik : ciks) {
                List<FilingRecord> filings = secEdgarClient.fetchRecentFilings(normalizeCik(cik), tickerForCik(cik), maxPerCik);
                for (FilingRecord record : filings) {
                    run.incrementFetched();
                    if (!isSupportedForm(record.formType())) {
                        run.incrementSkipped();
                        continue;
                    }

                    try {
                        UpsertResult result = upsertFiling(record);
                        if (result == UpsertResult.INSERTED) {
                            run.incrementInserted();
                        } else if (result == UpsertResult.UPDATED) {
                            run.incrementUpdated();
                        } else {
                            run.incrementSkipped();
                        }

                        if (includeDocuments && result.isInsertOrUpdate()) {
                            RawDocument raw = secEdgarClient.downloadDocument(record);
                            String storageUri = rawDocumentStorage.store(record.accessionNo(), raw);

                            FilingEntity persisted = filingRepository.findById(record.accessionNo())
                                .orElseThrow(() -> new IllegalStateException("Filing disappeared after save"));
                            persisted.setChecksum(raw.sha256());
                            persisted.setStorageUri(storageUri);
                            filingRepository.save(persisted);
                        }
                    } catch (Exception ex) {
                        run.incrementFailed();
                        ingestionFailureRepository.save(IngestionFailureEntity.of(
                            run.getRunId(),
                            record.accessionNo(),
                            "PROCESSING_ERROR",
                            truncate(ex.getMessage(), 400)
                        ));
                    }
                }
            }

            run.complete();
            ingestionRunRepository.save(run);
            return run.getRunId();
        } catch (Exception fatal) {
            run.fail(truncate(fatal.getMessage(), 400));
            ingestionRunRepository.save(run);
            throw fatal;
        }
    }

    public Optional<IngestionRunEntity> getRun(UUID runId) {
        return ingestionRunRepository.findById(runId);
    }

    public List<IngestionFailureEntity> getRunFailures(UUID runId) {
        return ingestionFailureRepository.findTop20ByRunIdOrderByCreatedAtDesc(runId);
    }

    public List<FilingEntity> searchFilings(String ticker, String formType, LocalDate from, LocalDate to, int limit) {
        return filingRepository.search(ticker, formType, from, to, PageRequest.of(0, Math.max(1, Math.min(limit, 200))));
    }

    private UpsertResult upsertFiling(FilingRecord record) {
        return filingRepository.findById(record.accessionNo())
            .map(existing -> {
                boolean changed = hasChanged(existing, record);
                if (!changed) {
                    return UpsertResult.SKIPPED;
                }
                existing.updateFrom(record);
                filingRepository.save(existing);
                return UpsertResult.UPDATED;
            })
            .orElseGet(() -> {
                filingRepository.save(FilingEntity.fromRecord(record));
                return UpsertResult.INSERTED;
            });
    }

    private boolean hasChanged(FilingEntity existing, FilingRecord incoming) {
        return !safe(existing.getSourceUrl()).equals(safe(incoming.sourceUrl()))
            || !safe(existing.getPrimaryDoc()).equals(safe(incoming.primaryDoc()))
            || !safe(existing.getFormType()).equals(safe(incoming.formType()))
            || !safe(existing.getTicker()).equals(safe(incoming.ticker()))
            || !safe(existing.getCompanyName()).equals(safe(incoming.companyName()))
            || !existing.getFilingDate().equals(incoming.filingDate());
    }

    private boolean isSupportedForm(String form) {
        return SUPPORTED_FORMS.contains(safe(form).toUpperCase(Locale.ROOT));
    }

    private String tickerForCik(String cik) {
        return TICKER_BY_CIK.getOrDefault(normalizeCik(cik), "UNK");
    }

    private String normalizeCik(String cik) {
        String digits = safe(cik).replaceAll("\\D", "");
        if (digits.isBlank()) {
            throw new IllegalArgumentException("Invalid CIK: " + cik);
        }
        long value = Long.parseLong(digits);
        return String.format("%010d", value);
    }

    private String safe(String value) {
        return value == null ? "" : value.trim();
    }

    private String truncate(String text, int max) {
        if (text == null || text.isBlank()) {
            return "unknown";
        }
        return text.length() <= max ? text : text.substring(0, max);
    }
}
