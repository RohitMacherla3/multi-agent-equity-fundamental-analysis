package com.ifip.ingestion.repository;

import com.ifip.ingestion.domain.FilingEntity;
import java.time.LocalDate;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface FilingRepository extends JpaRepository<FilingEntity, String> {

    @Query("""
        select f from FilingEntity f
        where (:ticker is null or upper(f.ticker) = upper(:ticker))
          and (:formType is null or upper(f.formType) = upper(:formType))
          and (:fromDate is null or f.filingDate >= :fromDate)
          and (:toDate is null or f.filingDate <= :toDate)
        order by f.filingDate desc
        """)
    List<FilingEntity> search(
        @Param("ticker") String ticker,
        @Param("formType") String formType,
        @Param("fromDate") LocalDate fromDate,
        @Param("toDate") LocalDate toDate,
        Pageable pageable
    );
}
