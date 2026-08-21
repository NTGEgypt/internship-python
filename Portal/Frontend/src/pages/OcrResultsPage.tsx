import { useState } from "react";
import { OcrResultsTable } from "@/components/OcrResultsTable";
import ViewOcrResultDialog from "@/components/ViewOcrResultDialog";
import { useOcrResults } from "../features/ocr-results/hooks/useOcrResults";
import type { OcrResult } from "../features/ocr-results/types/ocrResult";

export default function OcrResultsPage() {
  const {
    results,
    loading,
    error,
  } = useOcrResults();

  const [dialogOpen, setDialogOpen] = useState(false);

  const [selectedResult, setSelectedResult] =
    useState<OcrResult | null>(null);

  const handleRowClick = (result: OcrResult) => {
    setSelectedResult(result);
    setDialogOpen(true);
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>{error}</div>;
  }

  return (
    <div>
      <OcrResultsTable
        results={results}
        onRowClick={handleRowClick}
      />

      <ViewOcrResultDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        result={selectedResult}
      />
    </div>
  );
}