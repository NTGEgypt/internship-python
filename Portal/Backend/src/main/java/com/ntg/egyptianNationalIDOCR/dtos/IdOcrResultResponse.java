package com.ntg.egyptianNationalIDOCR.dtos;

import com.ntg.egyptianNationalIDOCR.entity.CaptureQuality;
import com.ntg.egyptianNationalIDOCR.entity.ReviewStatus;

import java.time.LocalDate;
import java.time.LocalDateTime;

public record IdOcrResultResponse(
        Long id,
        String originalFilename,
        String imageMimeType,
        String imageSha256,
        String nationalId,
        String fullName,
        String firstName,
        String lastName,
        LocalDate birthDate,
        String gender,
        String governorate,
        String address,
        String serialNumber,
        String ocrResult,
        CaptureQuality captureQuality,
        ReviewStatus reviewStatus,
        String reviewedBy,
        LocalDateTime reviewedAt,
        String decisionNote,
        String ocrModelVersion,
        LocalDateTime createdAt
) {
}