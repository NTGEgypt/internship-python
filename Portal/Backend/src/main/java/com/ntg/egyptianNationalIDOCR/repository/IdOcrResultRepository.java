package com.ntg.egyptianNationalIDOCR.repository;

import com.ntg.egyptianNationalIDOCR.entity.IdOcrResult;
import org.springframework.data.jpa.repository.JpaRepository;

public interface IdOcrResultRepository
        extends JpaRepository<IdOcrResult, Long> {
}