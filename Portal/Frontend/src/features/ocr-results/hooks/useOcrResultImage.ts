import { useEffect, useState } from "react";

type ImageResponse = {
  status: number;
  message: string;
  data: {
    mimeType: string;
    image: string;
  };
};

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

        const response = await fetch(
          `http://localhost:8080/api/ocr-results/${resultId}/image`
        );

        if (!response.ok) {
          throw new Error("Failed to retrieve image");
        }

        const result: ImageResponse = await response.json();

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