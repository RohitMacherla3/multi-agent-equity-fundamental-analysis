package com.ifip.ingestion.repository;

import com.ifip.ingestion.domain.IngestionRunEntity;
import java.util.UUID;
import org.springframework.data.jpa.repository.JpaRepository;

public interface IngestionRunRepository extends JpaRepository<IngestionRunEntity, UUID> {
}
