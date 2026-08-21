import { useState } from "react";

type ReviewResponse = {
  status: number;
  message: string;
  data: unknown;
};

export function useOcrReview() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const review = async (
    id: number,
    action: "approve" | "reject",
    decisionNote: string = ""
  ): Promise<ReviewResponse | null> => {
    setLoading(true);
    setError(null);

    try {
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

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "Something went wrong.");
      }

      return data;
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Something went wrong.";

      setError(message);
      return null;
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