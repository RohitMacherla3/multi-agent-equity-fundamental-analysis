package com.ifip.ingestion.client;

public interface RawDocumentStorage {

    String store(String accessionNo, RawDocument document);
}
