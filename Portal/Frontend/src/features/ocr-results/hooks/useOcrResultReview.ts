import { useState } from "react";

type ReviewAction = "approve" | "reject";

type ReviewResponse = {
  status: number;
  message: string;
  data: unknown;
};

export function useOcrResultReview() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const review = async (
    id: number,
    action: ReviewAction,
    decisionNote: string = ""
  ): Promise<ReviewResponse> => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `http://localhost:8080/api/ocr-results/${id}/${action}`,
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
        throw new Error(data.message || "Review operation failed");
      }

      return data;
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Review operation failed";

      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    review,
    loading,
    error,
  };
}