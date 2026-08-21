package com.ntg.egyptianNationalIDOCR.services;

import com.ntg.egyptianNationalIDOCR.dtos.ApiResponse;
import com.ntg.egyptianNationalIDOCR.dtos.IdOcrResultResponse;
import com.ntg.egyptianNationalIDOCR.dtos.ImageResponse;
import com.ntg.egyptianNationalIDOCR.dtos.ReviewRequest;
import com.ntg.egyptianNationalIDOCR.entity.IDOcrResult;
import com.ntg.egyptianNationalIDOCR.entity.ReviewStatus;
import com.ntg.egyptianNationalIDOCR.repository.IdOcrResultRepository;
import org.springframework.stereotype.Service;

import java.util.Base64;
import java.time.LocalDateTime;
import java.util.Optional;

@Service
public class IdOcrResultService {
    private final IdOcrResultRepository repository;

    public IdOcrResultService(IdOcrResultRepository repository) {
        this.repository = repository;
    }

    public ApiResponse getAllResults() {

        return new ApiResponse(
                200,
                "OK.",
                repository.findAll()
                        .stream()
                        .map(this::toResponse)
                        .toList()
        );
    }

    public ApiResponse getResultById(Long id) {

        Optional<IDOcrResult> optionalResult = repository.findById(id);

        if (optionalResult.isEmpty()) {
            return new ApiResponse(
                    404,
                    "Record doesn't exist.",
                    null
            );
        }

        IDOcrResult result = optionalResult.get();

        return new ApiResponse(
                200,
                "OK.",
                toResponse(result)
        );
    }

    public ApiResponse getImage(Long id) {

        IDOcrResult result = repository.findById(id)
                .orElse(null);

        if (result == null) {
            return new ApiResponse(
                    404,
                    "Record doesn't exist.",
                    null
            );
        }

        String base64Image = Base64.getEncoder()
                .encodeToString(result.getCardImage());

        ImageResponse imageResponse = new ImageResponse(
                result.getImageMimeType(),
                base64Image
        );

        return new ApiResponse(
                200,
                "Image retrieved successfully.",
                imageResponse
        );
    }

    public ApiResponse approve(Long id, ReviewRequest request) {

        IDOcrResult result = repository.findById(id)
                .orElse(null);

        if (result == null) {
            return new ApiResponse(
                    404,
                    "Record doesn't exist.",
                    null
            );
        }

        if (result.getReviewStatus() != ReviewStatus.PENDING) {
            return new ApiResponse(
                    409,
                    "Record has already been reviewed.",
                    null
            );
        }

        result.setReviewStatus(ReviewStatus.ACCEPTED);
        result.setReviewedBy("admin");
        result.setReviewedAt(LocalDateTime.now());
        result.setDecisionNote(request.getDecisionNote());

        IDOcrResult savedResult = repository.save(result);

        return new ApiResponse(
                200,
                "Record approved successfully.",
                toResponse(savedResult)
        );
    }

    public ApiResponse reject(Long id, ReviewRequest request) {

        IDOcrResult result = repository.findById(id)
                .orElse(null);

        if (result == null) {
            return new ApiResponse(
                    404,
                    "Record doesn't exist.",
                    null
            );
        }

        if (result.getReviewStatus() != ReviewStatus.PENDING) {
            return new ApiResponse(
                    409,
                    "Record has already been reviewed.",
                    null
            );
        }

        if (request.getDecisionNote() == null ||
                request.getDecisionNote().isBlank()) {

            return new ApiResponse(
                    400,
                    "A rejection reason is required.",
                    null
            );
        }

        result.setReviewStatus(ReviewStatus.REJECTED);
        result.setReviewedBy("admin");
        result.setReviewedAt(LocalDateTime.now());
        result.setDecisionNote(request.getDecisionNote());

        IDOcrResult savedResult = repository.save(result);

        return new ApiResponse(
                200,
                "Record rejected successfully.",
                toResponse(savedResult)
        );
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
