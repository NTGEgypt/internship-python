package com.ntg.ocr_mobile.data

import android.content.ContentResolver
import android.net.Uri
import android.util.Log
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException

class OcrRepository(
    private val contentResolver: ContentResolver,
    private val api: OcrApi = OcrApiProvider.api,
) {
    suspend fun processImage(uri: Uri): UploadResult {
        val mimeType = contentResolver.getType(uri)
            ?: return UploadResult.Failure("Could not identify the selected image.")
        if (mimeType !in supportedMimeTypes) {
            return UploadResult.Failure("Choose a JPEG or PNG image.")
        }

        val imageBytes = try {
            contentResolver.openInputStream(uri)?.use { it.readBytes() }
                ?: return UploadResult.Failure("Could not read the selected image.")
        } catch (_: IOException) {
            return UploadResult.Failure("Could not read the selected image.")
        }

        if (imageBytes.size > maxImageSizeBytes) {
            return UploadResult.Failure("The image must not exceed 15 MB.")
        }

        val requestBody = imageBytes.toRequestBody(mimeType.toMediaTypeOrNull())
        val imagePart = MultipartBody.Part.createFormData(
            name = "image",
            filename = "egyptian_id.${if (mimeType == "image/png") "png" else "jpg"}",
            body = requestBody,
        )

        return try {
            val response = api.processImage(imagePart)
            if (response.isSuccessful && response.body()?.success == true) {
                UploadResult.Success(response.body()?.submission_id)
            } else {
                UploadResult.Failure(errorMessage(response.errorBody()?.string()))
            }
        } catch (exception: IOException) {
            Log.e(TAG, "OCR API request failed", exception)
            UploadResult.Failure(
                "Could not reach the OCR server: ${exception.message ?: "unknown connection error"}",
            )
        } catch (_: Exception) {
            UploadResult.Failure("The image could not be processed. Please try again.")
        }
    }

    private fun errorMessage(errorBody: String?): String = try {
        JSONObject(errorBody.orEmpty()).optString("detail").ifBlank { defaultServerError }
    } catch (_: Exception) {
        defaultServerError
    }

    private companion object {
        const val TAG = "OcrRepository"
        val supportedMimeTypes = setOf("image/jpeg", "image/png")
        const val maxImageSizeBytes = 15 * 1024 * 1024
        const val defaultServerError = "The server could not process this image."
    }
}

sealed interface UploadResult {
    data class Success(val submissionId: Long?) : UploadResult
    data class Failure(val message: String) : UploadResult
}
