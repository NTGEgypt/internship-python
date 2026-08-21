import { useEffect, useState } from "react";
import { getOcrResults } from "../features/ocr-results/api/ocrResultsApi";
import type { OcrResult } from "../features/ocr-results/types/ocrResult";

export default function OcrResultsPage() {
  const [results, setResults] = useState<OcrResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadResults() {
      try {
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

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>{error}</div>;
  }

  return (
    <div>
      <h1>OCR Results</h1>

      <p>
        Total records: {results.length}
      </p>
    </div>
  );
}