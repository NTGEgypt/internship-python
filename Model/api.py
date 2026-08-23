"""HTTP API for the Egyptian National ID OCR pipeline.

Run locally with:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
import cv2
import numpy as np

from core.database import (
    DatabaseConfigurationError,
    DatabaseStorageError,
    save_ocr_result,
)
from core.pipeline import run_pipeline


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Egyptian National ID OCR API",
    version="1.0.0",
    description="Upload an Egyptian National ID image and receive its OCR result.",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report that the API process is available."""
    return {"status": "ok"}


@app.post("/api/v1/ocr/process")
async def process_id_image(
    image: UploadFile = File(
        ...,
        description="JPEG or PNG image of an Egyptian National ID card",
    ),
) -> dict:
    """Process an uploaded ID-card image and persist the OCR result."""

    request_started_at = time.perf_counter()

    # ---------------------------------------------------------
    # 1. Validate uploaded file type
    # ---------------------------------------------------------
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported.",
        )

    # ---------------------------------------------------------
    # 2. Read image bytes
    # ---------------------------------------------------------
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The uploaded image must not exceed 15 MB.",
        )

    # ---------------------------------------------------------
    # 3. Decode image using OpenCV
    # ---------------------------------------------------------
    decoded = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if decoded is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    logger.info("OCR request accepted; starting image processing.")

    # ---------------------------------------------------------
    # 4. Run the NEW OCR pipeline
    # ---------------------------------------------------------
    try:
        pipeline_result = run_pipeline(decoded)

        # PipelineResult -> dictionary
        # This dictionary is what the database and API response expect.
        result = pipeline_result.as_dict()
        result["capture_quality"] = "GOOD"

    except ValueError as exc:
        # Known pipeline rejections, such as:
        # - No card detected
        # - Invalid card
        # - No fields detected
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Don't expose internal model/pipeline errors to the client.
        logger.exception("OCR pipeline failed.")

        raise HTTPException(
            status_code=500,
            detail="Image processing failed.",
        ) from exc

    logger.info(
        "OCR processing completed in %.1f seconds; saving result to MySQL.",
        time.perf_counter() - request_started_at,
    )

    # ---------------------------------------------------------
    # 5. Save result to MySQL
    # ---------------------------------------------------------
    try:
        submission_id = save_ocr_result(
            result=result,
            image_bytes=image_bytes,
            original_filename=image.filename,
            image_mime_type=image.content_type,
        )

    except DatabaseConfigurationError as exc:
        logger.exception("Database storage is not configured.")

        raise HTTPException(
            status_code=503,
            detail="Result storage is not configured on the server.",
        ) from exc

    except DatabaseStorageError as exc:
        logger.exception("Could not save OCR submission to MySQL.")

        raise HTTPException(
            status_code=503,
            detail="Result storage is temporarily unavailable.",
        ) from exc

    # ---------------------------------------------------------
    # 6. Log successful submission
    # ---------------------------------------------------------
    logger.info(
        "OCR submission #%s saved in %.1f seconds total.",
        submission_id,
        time.perf_counter() - request_started_at,
    )

    # ---------------------------------------------------------
    # 7. Return JSON response
    # ---------------------------------------------------------
    return {
        "success": True,
        "submission_id": submission_id,
        "result": result,
    }