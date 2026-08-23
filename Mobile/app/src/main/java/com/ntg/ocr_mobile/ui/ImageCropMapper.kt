package com.ntg.ocr_mobile.ui

import android.graphics.Rect
import kotlin.math.max
import kotlin.math.roundToInt

object ImageCropMapper {

    fun mapGuideRectToImageRect(
        guideRectInPreview: Rect,
        previewViewWidth: Int,
        previewViewHeight: Int,
        uprightImageWidth: Int,
        uprightImageHeight: Int,
    ): Rect {

        require(previewViewWidth > 0) {
            "Preview width must be greater than zero."
        }

        require(previewViewHeight > 0) {
            "Preview height must be greater than zero."
        }

        require(uprightImageWidth > 0) {
            "Image width must be greater than zero."
        }

        require(uprightImageHeight > 0) {
            "Image height must be greater than zero."
        }

        /*
         * PreviewView.ScaleType.FILL_CENTER
         *
         * The image is scaled until it completely fills
         * the PreviewView.
         */
        val scale = max(
            previewViewWidth.toFloat() /
                    uprightImageWidth,

            previewViewHeight.toFloat() /
                    uprightImageHeight
        )

        /*
         * Size of the scaled image displayed in the preview.
         */
        val displayedWidth =
            uprightImageWidth * scale

        val displayedHeight =
            uprightImageHeight * scale

        /*
         * Because FILL_CENTER crops the excess,
         * calculate how much was cropped from each side.
         */
        val offsetX =
            (displayedWidth - previewViewWidth) / 2f

        val offsetY =
            (displayedHeight - previewViewHeight) / 2f

        /*
         * Convert PreviewView coordinates
         * back to image coordinates.
         */
        val left =
            ((guideRectInPreview.left + offsetX) / scale)
                .roundToInt()

        val top =
            ((guideRectInPreview.top + offsetY) / scale)
                .roundToInt()

        val right =
            ((guideRectInPreview.right + offsetX) / scale)
                .roundToInt()

        val bottom =
            ((guideRectInPreview.bottom + offsetY) / scale)
                .roundToInt()

        /*
         * Keep everything inside the bitmap.
         */
        return Rect(
            left.coerceIn(0, uprightImageWidth),
            top.coerceIn(0, uprightImageHeight),
            right.coerceIn(0, uprightImageWidth),
            bottom.coerceIn(0, uprightImageHeight)
        )
    }
}