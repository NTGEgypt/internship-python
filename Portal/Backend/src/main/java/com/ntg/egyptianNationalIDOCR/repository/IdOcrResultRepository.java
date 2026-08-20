package com.ntg.egyptianNationalIDOCR.repository;

import com.ntg.egyptianNationalIDOCR.entity.IDOcrResult;
import org.springframework.data.jpa.repository.JpaRepository;

public interface IdOcrResultRepository
        extends JpaRepository<IDOcrResult, Long> {
}