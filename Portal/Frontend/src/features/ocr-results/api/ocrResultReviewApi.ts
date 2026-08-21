import { API_BASE_URL } from "../../../lib/api";

export type ReviewAction = "approve" | "reject";

export type ReviewResponse = {
  status: number;
  message: string;
  data: unknown;
};

export async function reviewOcrResult(
  id: number,
  action: ReviewAction,
  decisionNote: string = ""
): Promise<ReviewResponse> {
  const response = await fetch(
    `${API_BASE_URL}/ocr-results/${id}/${action}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        decisionNote,
      }),
    }
  );

  const data: ReviewResponse = await response.json();

  if (!response.ok) {
    throw new Error(
      data.message || "Something went wrong."
    );
  }

  return data;
}