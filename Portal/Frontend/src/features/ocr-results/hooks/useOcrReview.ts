import { useState } from "react";
import {
  reviewOcrResult,
  type ReviewAction,
  type ReviewResponse,
} from "../api/ocrResultReviewApi";

export function useOcrReview() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const review = async (
    id: number,
    action: ReviewAction,
    decisionNote: string = ""
  ): Promise<ReviewResponse | null> => {
    setLoading(true);
    setError(null);

    try {
      const data = await reviewOcrResult(
        id,
        action,
        decisionNote
      );

      return data;
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Something went wrong.";

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