package com.ifip.ingestion.client;

public record RawDocument(String filename, byte[] bytes, String sha256) {
}
