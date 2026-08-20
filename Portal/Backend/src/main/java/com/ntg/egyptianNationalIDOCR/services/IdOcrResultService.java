package com.ntg.egyptianNationalIDOCR.services;

import com.ntg.egyptianNationalIDOCR.dtos.IdOcrResultResponse;
import com.ntg.egyptianNationalIDOCR.entity.IdOcrResult;
import com.ntg.egyptianNationalIDOCR.repository.IdOcrResultRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class IdOcrResultService {
    private final IdOcrResultRepository repository;

    public IdOcrResultService(IdOcrResultRepository repository) {
        this.repository = repository;
    }

    public List<IdOcrResultResponse> getAllResults() {

        return repository.findAll()
                .stream()
                .map(this::toResponse)
                .toList();
    }

    private IdOcrResultResponse toResponse(IdOcrResult result) {
        return new IdOcrResultResponse(
                result.getId(),
                result.getOriginalFilename(),
                result.getImageMimeType(),
                result.getImageSha256(),
                result.getNationalId(),
                result.getFullName(),
                result.getFirstName(),
                result.getLastName(),
                result.getBirthDate(),
                result.getGender(),
                result.getGovernorate(),
                result.getAddress(),
                result.getSerialNumber(),
                result.getOcrResult(),
                result.getCaptureQuality(),
                result.getReviewStatus(),
                result.getReviewedBy(),
                result.getReviewedAt(),
                result.getDecisionNote(),
                result.getOcrModelVersion(),
                result.getCreatedAt()
        );
    }
}
