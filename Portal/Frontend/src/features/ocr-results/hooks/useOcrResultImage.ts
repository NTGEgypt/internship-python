import { useEffect, useState } from "react";
import { getOcrResultImage } from "../api/ocrResultImageApi";

export function useOcrResultImage(
  resultId: number | null,
  enabled: boolean = true
) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!resultId || !enabled) {
      setImageUrl(null);
      return;
    }

    const fetchImage = async () => {
      try {
        setLoading(true);
        setError(null);

        const result = await getOcrResultImage(resultId);

        const { mimeType, image } = result.data;

        setImageUrl(`data:${mimeType};base64,${image}`);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to retrieve image"
        );

        setImageUrl(null);
      } finally {
        setLoading(false);
      }
    };

    fetchImage();
  }, [resultId, enabled]);

  return {
    imageUrl,
    loading,
    error,
  };
}