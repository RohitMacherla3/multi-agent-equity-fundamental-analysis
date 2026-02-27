package com.ifip.ingestion.client;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public class LocalRawDocumentStorage implements RawDocumentStorage {

    private final Path basePath;

    public LocalRawDocumentStorage(Path basePath) {
        this.basePath = basePath;
    }

    @Override
    public String store(String accessionNo, RawDocument document) {
        try {
            Files.createDirectories(basePath);
            String safeFileName = accessionNo + "__" + document.filename().replace("/", "_");
            Path target = basePath.resolve(safeFileName);
            Files.write(target, document.bytes(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
            return target.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to store raw filing document", e);
        }
    }

}
