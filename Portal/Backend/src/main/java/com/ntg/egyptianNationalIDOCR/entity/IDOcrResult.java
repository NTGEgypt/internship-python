package com.ntg.egyptianNationalIDOCR.entity;

import jakarta.persistence.*;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "id_ocr_results")
public class IdOcrResult {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "original_filename")
    private String originalFilename;

    @Column(name = "image_mime_type", nullable = false)
    private String imageMimeType;

    @Column(name = "image_sha256", nullable = false, length = 64)
    private String imageSha256;

    @Lob
    @Column(name = "card_image", nullable = false)
    private byte[] cardImage;

    @Column(name = "national_id", length = 14)
    private String nationalId;

    @Column(name = "full_name")
    private String fullName;

    @Column(name = "first_name")
    private String firstName;

    @Column(name = "last_name")
    private String lastName;

    @Column(name = "birth_date")
    private LocalDate birthDate;

    private String gender;

    private String governorate;

    @Column(columnDefinition = "TEXT")
    private String address;

    @Column(name = "serial_number")
    private String serialNumber;

    @Column(name = "ocr_result", columnDefinition = "JSON")
    private String ocrResult;

    @Enumerated(EnumType.STRING)
    @Column(name = "capture_quality")
    private CaptureQuality captureQuality;

    @Enumerated(EnumType.STRING)
    @Column(name = "review_status", nullable = false)
    private ReviewStatus reviewStatus;

    @Column(name = "reviewed_by")
    private String reviewedBy;

    @Column(name = "reviewed_at")
    private LocalDateTime reviewedAt;

    @Column(name = "decision_note", columnDefinition = "TEXT")
    private String decisionNote;

    @Column(name = "ocr_model_version")
    private String ocrModelVersion;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    // Getters and setters

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getOriginalFilename() {
        return originalFilename;
    }

    public void setOriginalFilename(String originalFilename) {
        this.originalFilename = originalFilename;
    }

    public String getImageMimeType() {
        return imageMimeType;
    }

    public void setImageMimeType(String imageMimeType) {
        this.imageMimeType = imageMimeType;
    }

    public String getImageSha256() {
        return imageSha256;
    }

    public void setImageSha256(String imageSha256) {
        this.imageSha256 = imageSha256;
    }

    public byte[] getCardImage() {
        return cardImage;
    }

    public void setCardImage(byte[] cardImage) {
        this.cardImage = cardImage;
    }

    public String getNationalId() {
        return nationalId;
    }

    public void setNationalId(String nationalId) {
        this.nationalId = nationalId;
    }

    public String getFullName() {
        return fullName;
    }

    public void setFullName(String fullName) {
        this.fullName = fullName;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public LocalDate getBirthDate() {
        return birthDate;
    }

    public void setBirthDate(LocalDate birthDate) {
        this.birthDate = birthDate;
    }

    public String getGovernorate() {
        return governorate;
    }

    public void setGovernorate(String governorate) {
        this.governorate = governorate;
    }

    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getOcrResult() {
        return ocrResult;
    }

    public void setOcrResult(String ocrResult) {
        this.ocrResult = ocrResult;
    }

    public String getSerialNumber() {
        return serialNumber;
    }

    public void setSerialNumber(String serialNumber) {
        this.serialNumber = serialNumber;
    }

    public CaptureQuality getCaptureQuality() {
        return captureQuality;
    }

    public void setCaptureQuality(CaptureQuality captureQuality) {
        this.captureQuality = captureQuality;
    }

    public ReviewStatus getReviewStatus() {
        return reviewStatus;
    }

    public void setReviewStatus(ReviewStatus reviewStatus) {
        this.reviewStatus = reviewStatus;
    }

    public LocalDateTime getReviewedAt() {
        return reviewedAt;
    }

    public void setReviewedAt(LocalDateTime reviewedAt) {
        this.reviewedAt = reviewedAt;
    }

    public String getReviewedBy() {
        return reviewedBy;
    }

    public void setReviewedBy(String reviewedBy) {
        this.reviewedBy = reviewedBy;
    }

    public String getOcrModelVersion() {
        return ocrModelVersion;
    }

    public void setOcrModelVersion(String ocrModelVersion) {
        this.ocrModelVersion = ocrModelVersion;
    }

    public String getDecisionNote() {
        return decisionNote;
    }

    public void setDecisionNote(String decisionNote) {
        this.decisionNote = decisionNote;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
}

public enum CaptureQuality {
    GOOD,
    BORDERLINE,
    BAD
}

public enum ReviewStatus {
    PENDING,
    ACCEPTED,
    REJECTED
}
