import { API_BASE_URL } from "../../../lib/api";

export type ImageResponse = {
  status: number;
  message: string;
  data: {
    mimeType: string;
    image: string;
  };
};

export async function getOcrResultImage(
  resultId: number
): Promise<ImageResponse> {
  const response = await fetch(
    `${API_BASE_URL}/ocr-results/${resultId}/image`
  );

  if (!response.ok) {
    throw new Error("Failed to retrieve image");
  }

  return response.json();
}