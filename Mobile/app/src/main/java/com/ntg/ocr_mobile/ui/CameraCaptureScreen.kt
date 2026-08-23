package com.ntg.ocr_mobile.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.net.Uri
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.boundsInParent
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private const val ID_CARD_ASPECT_RATIO = 85.60f / 53.98f

@Composable
fun CameraCaptureScreen(
    onPhotoCaptured: (Uri) -> Unit,
    onClose: () -> Unit,
    onError: (String) -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    /*
     * Main-thread executor.
     * Used for CameraX callbacks and UI updates.
     */
    val mainExecutor = remember(context) {
        ContextCompat.getMainExecutor(context)
    }

    /*
     * Background executor.
     *
     * Bitmap decoding, cropping and JPEG compression
     * should NOT happen on the main thread.
     */
    val processingExecutor: ExecutorService = remember {
        Executors.newSingleThreadExecutor()
    }

    /*
     * Camera preview.
     */
    val previewView = remember(context) {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode =
                PreviewView.ImplementationMode.COMPATIBLE
        }
    }

    var imageCapture by remember {
        mutableStateOf<ImageCapture?>(null)
    }

    var cameraProvider by remember {
        mutableStateOf<ProcessCameraProvider?>(null)
    }

    var isCapturing by remember {
        mutableStateOf(false)
    }

    var guideRectPx by remember {
        mutableStateOf<Rect?>(null)
    }

    var previewWidth by remember {
        mutableStateOf(0)
    }

    var previewHeight by remember {
        mutableStateOf(0)
    }

    /*
     * =========================
     * CAMERA SETUP
     * =========================
     */

    DisposableEffect(lifecycleOwner) {

        val providerFuture =
            ProcessCameraProvider.getInstance(context)

        providerFuture.addListener({

            try {

                val provider =
                    providerFuture.get()

                val preview =
                    Preview.Builder()
                        .build()
                        .also {
                            it.setSurfaceProvider(
                                previewView.surfaceProvider
                            )
                        }

                val capture =
                    ImageCapture.Builder()
                        .setCaptureMode(
                            ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY
                        )
                        .build()

                provider.unbindAll()

                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    capture
                )

                cameraProvider = provider
                imageCapture = capture

            } catch (e: Exception) {

                e.printStackTrace()

                onError(
                    "Could not open camera: ${
                        e.message ?: "unknown error"
                    }"
                )
            }

        }, mainExecutor)

        onDispose {

            cameraProvider?.unbindAll()

            processingExecutor.shutdown()
        }
    }

    /*
     * =========================
     * SCREEN
     * =========================
     */

    Box(
        modifier = Modifier
            .fillMaxSize()
            .onGloballyPositioned { coordinates ->

                previewWidth = coordinates.size.width
                previewHeight = coordinates.size.height
            }
    ) {

        /*
         * Camera preview
         */

        AndroidView(
            factory = {
                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        /*
         * Guide border
         */

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp)
                .aspectRatio(ID_CARD_ASPECT_RATIO)
                .align(Alignment.Center)
                .clip(MaterialTheme.shapes.medium)
                .border(
                    width = 3.dp,
                    color = Color.White,
                    shape = MaterialTheme.shapes.medium
                )
                .onGloballyPositioned { coordinates ->

                    val bounds =
                        coordinates.boundsInParent()

                    guideRectPx = Rect(
                        bounds.left.toInt(),
                        bounds.top.toInt(),
                        bounds.right.toInt(),
                        bounds.bottom.toInt()
                    )
                }
        )

        /*
         * Bottom controls
         */

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),

            horizontalAlignment =
                Alignment.CenterHorizontally,

            verticalArrangement =
                Arrangement.Bottom
        ) {

            Button(
                onClick = {

                    val capture =
                        imageCapture
                            ?: return@Button

                    val guideRect =
                        guideRectPx
                            ?: return@Button

                    if (isCapturing) {
                        return@Button
                    }

                    if (
                        previewWidth <= 0 ||
                        previewHeight <= 0
                    ) {
                        onError(
                            "Camera preview is not ready."
                        )
                        return@Button
                    }

                    isCapturing = true

                    /*
                     * =========================
                     * ORIGINAL FILE
                     * =========================
                     */

                    val originalFile =
                        File(
                            context.cacheDir,
                            "original_${
                                System.currentTimeMillis()
                            }.jpg"
                        )

                    val outputOptions =
                        ImageCapture
                            .OutputFileOptions
                            .Builder(originalFile)
                            .build()

                    /*
                     * =========================
                     * TAKE PHOTO
                     * =========================
                     */

                    capture.takePicture(
                        outputOptions,
                        mainExecutor,

                        object :
                            ImageCapture.OnImageSavedCallback {

                            override fun onImageSaved(
                                outputFileResults:
                                ImageCapture.OutputFileResults
                            ) {

                                /*
                                 * Camera capture succeeded.
                                 *
                                 * Now move bitmap processing
                                 * to the background thread.
                                 */

                                processingExecutor.execute {

                                    var originalBitmap:
                                            Bitmap? = null

                                    var croppedBitmap:
                                            Bitmap? = null

                                    try {

                                        /*
                                         * =========================
                                         * DECODE ORIGINAL
                                         * =========================
                                         */

                                        originalBitmap =
                                            BitmapFactory.decodeFile(
                                                originalFile.absolutePath
                                            )

                                        if (
                                            originalBitmap == null
                                        ) {
                                            throw Exception(
                                                "Could not decode captured image."
                                            )
                                        }

                                        /*
                                         * =========================
                                         * MAP GUIDE → IMAGE
                                         * =========================
                                         */

                                        val cropRect =
                                            ImageCropMapper
                                                .mapGuideRectToImageRect(
                                                    guideRectInPreview =
                                                        guideRect,

                                                    previewViewWidth =
                                                        previewWidth,

                                                    previewViewHeight =
                                                        previewHeight,

                                                    uprightImageWidth =
                                                        originalBitmap.width,

                                                    uprightImageHeight =
                                                        originalBitmap.height
                                                )

                                        /*
                                         * =========================
                                         * VALIDATE CROP
                                         * =========================
                                         */

                                        if (
                                            cropRect.left < 0 ||
                                            cropRect.top < 0 ||
                                            cropRect.right >
                                            originalBitmap.width ||
                                            cropRect.bottom >
                                            originalBitmap.height
                                        ) {
                                            throw Exception(
                                                "Crop rectangle is outside image bounds: $cropRect"
                                            )
                                        }

                                        if (
                                            cropRect.width() <= 0 ||
                                            cropRect.height() <= 0
                                        ) {
                                            throw Exception(
                                                "Invalid crop area: $cropRect"
                                            )
                                        }

                                        /*
                                         * =========================
                                         * CROP
                                         * =========================
                                         */

                                        croppedBitmap =
                                            Bitmap.createBitmap(
                                                originalBitmap,
                                                cropRect.left,
                                                cropRect.top,
                                                cropRect.width(),
                                                cropRect.height()
                                            )

                                        /*
                                         * =========================
                                         * SAVE CROPPED FILE
                                         * =========================
                                         */

                                        val croppedFile =
                                            File(
                                                context.cacheDir,
                                                "egyptian_id_${
                                                    System.currentTimeMillis()
                                                }.jpg"
                                            )

                                        FileOutputStream(
                                            croppedFile
                                        ).use { output ->

                                            val success =
                                                croppedBitmap.compress(
                                                    Bitmap.CompressFormat.JPEG,
                                                    95,
                                                    output
                                                )

                                            if (!success) {
                                                throw Exception(
                                                    "Failed to compress cropped image."
                                                )
                                            }
                                        }

                                        /*
                                         * =========================
                                         * SUCCESS
                                         * =========================
                                         */

                                        mainExecutor.execute {

                                            isCapturing = false

                                            onPhotoCaptured(
                                                Uri.fromFile(
                                                    croppedFile
                                                )
                                            )
                                        }

                                    } catch (e: Throwable) {

                                        /*
                                         * Throwable instead of Exception
                                         * so we also catch serious bitmap
                                         * errors such as memory allocation
                                         * problems during debugging.
                                         */

                                        e.printStackTrace()

                                        mainExecutor.execute {

                                            isCapturing = false

                                            onError(
                                                "Could not process photo: ${
                                                    e.message
                                                        ?: e.javaClass.simpleName
                                                }"
                                            )
                                        }

                                    } finally {

                                        /*
                                         * Free bitmap memory.
                                         */

                                        originalBitmap?.recycle()

                                        if (
                                            croppedBitmap != null &&
                                            croppedBitmap !== originalBitmap
                                        ) {
                                            croppedBitmap.recycle()
                                        }
                                    }
                                }
                            }

                            override fun onError(
                                exception:
                                ImageCaptureException
                            ) {

                                exception.printStackTrace()

                                isCapturing = false

                                onError(
                                    "Could not take photo: ${
                                        exception.message
                                            ?: "unknown camera error"
                                    }"
                                )
                            }
                        }
                    )
                },

                enabled =
                    imageCapture != null &&
                            !isCapturing,

                modifier =
                    Modifier.size(
                        width = 180.dp,
                        height = 52.dp
                    )
            ) {

                Text(
                    if (isCapturing) {
                        "Processing..."
                    } else {
                        "Take photo"
                    }
                )
            }

            Spacer(
                modifier = Modifier.size(8.dp)
            )

            OutlinedButton(
                onClick = onClose,
                enabled = !isCapturing
            ) {
                Text("Cancel")
            }
        }

        /*
         * Top instructions.
         */

        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(24.dp),

            horizontalAlignment =
                Alignment.CenterHorizontally
        ) {

            Text(
                text = "Align the ID card inside the frame",
                color = Color.White,
                style = MaterialTheme.typography.titleMedium,
                textAlign = TextAlign.Center
            )

            Text(
                text = "Keep all corners visible and avoid glare.",
                color = Color.White,
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center
            )
        }
    }
}