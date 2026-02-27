package com.ifip.ingestion.repository;

import com.ifip.ingestion.domain.IngestionFailureEntity;
import java.util.List;
import java.util.UUID;
import org.springframework.data.jpa.repository.JpaRepository;

public interface IngestionFailureRepository extends JpaRepository<IngestionFailureEntity, UUID> {

    List<IngestionFailureEntity> findTop20ByRunIdOrderByCreatedAtDesc(UUID runId);
}
