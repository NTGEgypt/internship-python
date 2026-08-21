import { useEffect, useState } from "react";
import { getOcrResults } from "../features/ocr-results/api/ocrResultsApi";
import type { OcrResult } from "../features/ocr-results/types/ocrResult";
import { OcrResultsTable } from "@/components/OcrResultsTable";
import ViewOcrResultDialog from "@/components/ViewOcrResultDialog";

export default function OcrResultsPage() {
  const [results, setResults] = useState<OcrResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedResult, setSelectedResult] =
    useState<OcrResult | null>(null);

  const handleRowClick = (result: OcrResult) => {
    setSelectedResult(result);
    setDialogOpen(true);
  };

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
      <OcrResultsTable results={results} onRowClick={handleRowClick} />

      <ViewOcrResultDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        result={selectedResult}
      />
    </div>
  );
}
