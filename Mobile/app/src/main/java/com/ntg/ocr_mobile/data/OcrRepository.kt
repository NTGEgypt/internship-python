package com.ntg.ocr_mobile.data

import android.content.ContentResolver
import android.net.Uri
import android.util.Log
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.IOException

class OcrRepository(
    private val contentResolver: ContentResolver,
    private val api: OcrApi = OcrApiProvider.api,
) {

    suspend fun processImage(uri: Uri): UploadResult {

        /*
         * =========================
         * GET MIME TYPE
         * =========================
         *
         * Gallery images usually have content:// URIs.
         * Camera images currently have file:// URIs.
         */
        val mimeType = when (uri.scheme) {

            "content" -> {
                contentResolver.getType(uri)
            }

            "file" -> {
                when {
                    uri.path?.endsWith(
                        ".png",
                        ignoreCase = true
                    ) == true -> "image/png"

                    uri.path?.endsWith(
                        ".jpg",
                        ignoreCase = true
                    ) == true ||
                            uri.path?.endsWith(
                                ".jpeg",
                                ignoreCase = true
                            ) == true -> "image/jpeg"

                    else -> null
                }
            }

            else -> null
        }

        if (mimeType == null) {
            return UploadResult.Failure(
                "Could not identify the selected image."
            )
        }

        /*
         * =========================
         * CHECK IMAGE TYPE
         * =========================
         */

        if (mimeType !in supportedMimeTypes) {
            return UploadResult.Failure(
                "Choose a JPEG or PNG image."
            )
        }

        /*
         * =========================
         * READ IMAGE
         * =========================
         */

        val imageBytes = try {

            when (uri.scheme) {

                /*
                 * Gallery / MediaStore
                 */
                "content" -> {
                    contentResolver
                        .openInputStream(uri)
                        ?.use { it.readBytes() }
                        ?: return UploadResult.Failure(
                            "Could not read the selected image."
                        )
                }

                /*
                 * Camera-generated file
                 */
                "file" -> {
                    val path = uri.path
                        ?: return UploadResult.Failure(
                            "Could not read the selected image."
                        )

                    File(path).readBytes()
                }

                else -> {
                    return UploadResult.Failure(
                        "Unsupported image URI."
                    )
                }
            }

        } catch (_: IOException) {

            return UploadResult.Failure(
                "Could not read the selected image."
            )

        } catch (_: SecurityException) {

            return UploadResult.Failure(
                "Permission denied while reading the selected image."
            )
        }

        /*
         * =========================
         * CHECK FILE SIZE
         * =========================
         */

        if (imageBytes.size > maxImageSizeBytes) {
            return UploadResult.Failure(
                "The image must not exceed 15 MB."
            )
        }

        /*
         * =========================
         * CREATE MULTIPART REQUEST
         * =========================
         */

        val requestBody =
            imageBytes.toRequestBody(
                mimeType.toMediaTypeOrNull()
            )

        val extension =
            if (mimeType == "image/png") {
                "png"
            } else {
                "jpg"
            }

        val imagePart =
            MultipartBody.Part.createFormData(
                name = "image",
                filename = "egyptian_id.$extension",
                body = requestBody,
            )

        /*
         * =========================
         * SEND TO OCR SERVER
         * =========================
         */

        return try {

            val response =
                api.processImage(imagePart)

            if (
                response.isSuccessful &&
                response.body()?.success == true
            ) {

                UploadResult.Success(
                    response.body()?.submission_id
                )

            } else {

                UploadResult.Failure(
                    errorMessage(
                        response.errorBody()?.string()
                    )
                )
            }

        } catch (exception: IOException) {

            Log.e(
                TAG,
                "OCR API request failed",
                exception
            )

            UploadResult.Failure(
                "Could not reach the OCR server: ${
                    exception.message
                        ?: "unknown connection error"
                }"
            )

        } catch (exception: Exception) {

            Log.e(
                TAG,
                "Unexpected OCR processing error",
                exception
            )

            UploadResult.Failure(
                "The image could not be processed. Please try again."
            )
        }
    }

    private fun errorMessage(
        errorBody: String?
    ): String =
        try {

            JSONObject(
                errorBody.orEmpty()
            )
                .optString("detail")
                .ifBlank {
                    defaultServerError
                }

        } catch (_: Exception) {

            defaultServerError
        }

    private companion object {

        const val TAG = "OcrRepository"

        val supportedMimeTypes =
            setOf(
                "image/jpeg",
                "image/png"
            )

        const val maxImageSizeBytes =
            15 * 1024 * 1024

        const val defaultServerError =
            "The server could not process this image."
    }
}

sealed interface UploadResult {

    data class Success(
        val submissionId: Long?
    ) : UploadResult

    data class Failure(
        val message: String
    ) : UploadResult
}