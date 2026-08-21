export type CaptureQuality =
  | "GOOD"
  | "BORDERLINE"
  | "BAD";

export type ReviewStatus =
  | "PENDING"
  | "ACCEPTED"
  | "REJECTED";

export interface OcrResult {
  id: number;

  originalFilename: string | null;
  imageMimeType: string;
  imageSha256: string;

  nationalId: string | null;
  fullName: string | null;
  firstName: string | null;
  lastName: string | null;
  birthDate: string | null;
  gender: string | null;
  governorate: string | null;
  address: string | null;
  serialNumber: string | null;

  ocrResult: string;

  captureQuality: CaptureQuality | null;

  reviewStatus: ReviewStatus;
  reviewedBy: string | null;
  reviewedAt: string | null;
  decisionNote: string | null;

  ocrModelVersion: string | null;
  createdAt: string;
}

export interface ApiResponse<T> {
  status: number;
  message: string;
  data: T;
}

export type OcrResultsResponse = ApiResponse<OcrResult[]>;