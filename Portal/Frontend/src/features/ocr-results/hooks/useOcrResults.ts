import { useEffect, useState } from "react";
import { getOcrResults } from "../api/ocrResultsApi";
import type { OcrResult } from "../types/ocrResult";

export function useOcrResults() {
  const [results, setResults] = useState<OcrResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadResults() {
      try {
        setLoading(true);
        setError(null);

        const data = await getOcrResults();

        setResults(data);
      } catch {
        setError("Failed to load OCR results");
      } finally {
        setLoading(false);
      }
    }

    loadResults();
  }, []);

  return {
    results,
    loading,
    error,
  };
}