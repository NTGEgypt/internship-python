import type {
  ApiResponse,
  OcrResult,
} from "../types/ocrResult";
import { API_BASE_URL } from "../../../lib/api";

export async function getOcrResults(): Promise<OcrResult[]> {
  const response = await fetch(
    `${API_BASE_URL}/ocr-results`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch OCR results");
  }

  const result: ApiResponse<OcrResult[]> =
    await response.json();

  return result.data;
}