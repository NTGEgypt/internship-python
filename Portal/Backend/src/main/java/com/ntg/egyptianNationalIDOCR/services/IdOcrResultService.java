package com.ntg.egyptianNationalIDOCR.services;

import com.ntg.egyptianNationalIDOCR.dtos.IdOcrResultResponse;
import com.ntg.egyptianNationalIDOCR.dtos.ReviewRequest;
import com.ntg.egyptianNationalIDOCR.entity.IDOcrResult;
import com.ntg.egyptianNationalIDOCR.entity.ReviewStatus;
import com.ntg.egyptianNationalIDOCR.repository.IdOcrResultRepository;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.time.LocalDateTime;
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

    public IdOcrResultResponse getResultById(Long id) {

        IDOcrResult result = repository.findById(id)
                .orElseThrow(() ->
                        new RuntimeException("OCR result not found with id: " + id)
                );

        return toResponse(result);
    }

    public ResponseEntity<byte[]> getImage(Long id) {

        IDOcrResult result = repository.findById(id)
                .orElseThrow(() ->
                        new RuntimeException("OCR result not found with id: " + id)
                );

        MediaType mediaType = MediaType.parseMediaType(
                result.getImageMimeType()
        );

        return ResponseEntity
                .ok()
                .contentType(mediaType)
                .body(result.getCardImage());
    }

    public IdOcrResultResponse approve(Long id, ReviewRequest request) {

        IDOcrResult result = repository.findById(id)
                .orElseThrow(() ->
                        new RuntimeException(
                                "OCR result not found with id: " + id
                        )
                );

        if (result.getReviewStatus() != ReviewStatus.PENDING) {
            throw new ResponseStatusException(
                    HttpStatus.CONFLICT,
                    "Record has already been reviewed"
            );
        }

        result.setReviewStatus(ReviewStatus.ACCEPTED);

        result.setReviewedBy("admin");

        result.setReviewedAt(LocalDateTime.now());

        result.setDecisionNote(request.getDecisionNote());

        IDOcrResult savedResult = repository.save(result);

        return toResponse(savedResult);
    }

    public IdOcrResultResponse reject(Long id, ReviewRequest request) {

        IDOcrResult result = repository.findById(id)
                .orElseThrow(() ->
                        new RuntimeException("OCR result not found with id: " + id)
                );

        if (result.getReviewStatus() != ReviewStatus.PENDING) {
            throw new ResponseStatusException(
                    HttpStatus.CONFLICT,
                    "Record has already been reviewed"
            );
        }

        if (request.getDecisionNote() == null ||
                request.getDecisionNote().isBlank()) {

            throw new ResponseStatusException(
                    HttpStatus.BAD_REQUEST,
                    "A rejection reason is required"
            );
        }

        result.setReviewStatus(ReviewStatus.REJECTED);
        result.setReviewedBy("admin");
        result.setReviewedAt(LocalDateTime.now());
        result.setDecisionNote(request.getDecisionNote());

        IDOcrResult savedResult = repository.save(result);

        return toResponse(savedResult);
    }

    private IdOcrResultResponse toResponse(IDOcrResult result) {
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
