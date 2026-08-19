package com.ntg.ocr_mobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.material3.Surface
import com.ntg.ocr_mobile.ui.OcrUploadScreen
import com.ntg.ocr_mobile.ui.theme.OCRMobileTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            OCRMobileTheme {
                Surface {
                    OcrUploadScreen()
                }
            }
        }
    }
}
