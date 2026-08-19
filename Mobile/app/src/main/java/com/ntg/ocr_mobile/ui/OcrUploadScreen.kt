package com.ntg.ocr_mobile.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.core.content.ContextCompat
import androidx.compose.ui.platform.LocalContext
import coil.compose.AsyncImage

@Composable
fun OcrUploadScreen(viewModel: OcrUploadViewModel = viewModel()) {
    val context = LocalContext.current
    val state = viewModel.uiState
    var isCameraOpen by remember { mutableStateOf(false) }
    val cameraPermission = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted ->
            if (granted) {
                isCameraOpen = true
            } else {
                viewModel.showError("Camera permission is required to take an ID photo.")
            }
        },
    )
    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri -> uri?.let(viewModel::selectImage) },
    )

    if (isCameraOpen) {
        CameraCaptureScreen(
            onPhotoCaptured = { uri ->
                isCameraOpen = false
                viewModel.selectImage(uri)
            },
            onClose = { isCameraOpen = false },
            onError = { message ->
                isCameraOpen = false
                viewModel.showError(message)
            },
        )
        return
    }

    fun openCamera() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            isCameraOpen = true
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 24.dp, vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Egyptian ID OCR",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = "Choose a clear image of an Egyptian National ID card to process it securely.",
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(28.dp))

        if (state.selectedImage == null) {
            Button(
                onClick = ::openCamera,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Take photo")
            }
            Spacer(Modifier.height(10.dp))
            OutlinedButton(
                onClick = {
                    imagePicker.launch(PickVisualMediaRequest(PickVisualMedia.ImageOnly))
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = !state.isProcessing,
            ) {
                Text("Choose image from storage")
            }
        } else {
            Card(modifier = Modifier.fillMaxWidth()) {
                AsyncImage(
                    model = state.selectedImage,
                    contentDescription = "Selected ID image",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(230.dp),
                    contentScale = ContentScale.Crop,
                )
            }
            Spacer(Modifier.height(16.dp))

            if (state.isProcessing) {
                CircularProgressIndicator(modifier = Modifier.size(40.dp))
                Spacer(Modifier.height(12.dp))
                Text("Processing your ID. This may take a couple of minutes…")
            } else {
                Button(
                    onClick = viewModel::processSelectedImage,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Process image")
                }
                Spacer(Modifier.height(8.dp))
                OutlinedButton(
                    onClick = {
                        imagePicker.launch(PickVisualMediaRequest(PickVisualMedia.ImageOnly))
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Choose a different image")
                }
            }
        }

        state.outcome?.let { outcome ->
            Spacer(Modifier.height(20.dp))
            OutcomeCard(outcome = outcome, onDismiss = viewModel::clearSelection)
        }
    }
}

@Composable
private fun OutcomeCard(outcome: UploadOutcome, onDismiss: () -> Unit) {
    val isSuccess = outcome is UploadOutcome.Success
    val containerColor = if (isSuccess) {
        MaterialTheme.colorScheme.primaryContainer
    } else {
        MaterialTheme.colorScheme.errorContainer
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = containerColor),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = if (isSuccess) "Processing complete" else "Processing failed",
                fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = when (outcome) {
                    is UploadOutcome.Success -> outcome.submissionId?.let {
                        "Your ID was processed and saved successfully. Submission #$it."
                    } ?: "Your ID was processed successfully."
                    is UploadOutcome.Error -> outcome.message
                },
            )
            Spacer(Modifier.height(12.dp))
            OutlinedButton(onClick = onDismiss) {
                Text("Process another image")
            }
        }
    }
}
