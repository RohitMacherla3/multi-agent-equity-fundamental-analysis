package com.ifip.ingestion.controller;

import com.ifip.ingestion.service.IngestionJobService;
import java.time.LocalDate;
import java.util.List;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/v1/filings")
public class FilingsController {

    private final IngestionJobService ingestionJobService;

    public FilingsController(IngestionJobService ingestionJobService) {
        this.ingestionJobService = ingestionJobService;
    }

    @GetMapping
    public List<FilingResponse> list(
        @RequestParam(required = false) String ticker,
        @RequestParam(required = false) String formType,
        @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
        @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
        @RequestParam(defaultValue = "50") int limit
    ) {
        return ingestionJobService.searchFilings(ticker, formType, from, to, limit)
            .stream()
            .map(FilingResponse::from)
            .toList();
    }
}
