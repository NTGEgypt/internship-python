package com.ntg.ocr_mobile.ui

import android.app.Application
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ntg.ocr_mobile.data.OcrRepository
import com.ntg.ocr_mobile.data.UploadResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class OcrUploadUiState(
    val selectedImage: Uri? = null,
    val isProcessing: Boolean = false,
    val outcome: UploadOutcome? = null,
)

sealed interface UploadOutcome {
    data class Success(val submissionId: Long?) : UploadOutcome
    data class Error(val message: String) : UploadOutcome
}

class OcrUploadViewModel(application: Application) : AndroidViewModel(application) {
    private val repository = OcrRepository(application.contentResolver)

    var uiState by mutableStateOf(OcrUploadUiState())
        private set

    fun selectImage(uri: Uri) {
        uiState = OcrUploadUiState(selectedImage = uri)
    }

    fun processSelectedImage() {
        val imageUri = uiState.selectedImage ?: return
        if (uiState.isProcessing) return

        uiState = uiState.copy(isProcessing = true, outcome = null)
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                repository.processImage(imageUri)
            }
            uiState = when (result) {
                is UploadResult.Success -> OcrUploadUiState(
                    selectedImage = imageUri,
                    outcome = UploadOutcome.Success(result.submissionId),
                )
                is UploadResult.Failure -> OcrUploadUiState(
                    selectedImage = imageUri,
                    outcome = UploadOutcome.Error(result.message),
                )
            }
        }
    }

    fun clearSelection() {
        uiState = OcrUploadUiState()
    }

    fun showError(message: String) {
        uiState = uiState.copy(outcome = UploadOutcome.Error(message))
    }
}
