package com.ntg.ocr_mobile.data

import com.google.gson.JsonObject
import com.ntg.ocr_mobile.BuildConfig
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import java.util.concurrent.TimeUnit

interface OcrApi {
    @Multipart
    @POST("api/v1/ocr/process")
    suspend fun processImage(@Part image: MultipartBody.Part): Response<OcrProcessResponse>
}

data class OcrProcessResponse(
    val success: Boolean,
    val submission_id: Long? = null,
    val result: JsonObject? = null,
)

object OcrApiProvider {
    private val httpClient = OkHttpClient.Builder()
        // OCR normally takes one to two minutes. OkHttp's default read timeout
        // is only 10 seconds, so each relevant timeout must be increased.
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(1, TimeUnit.MINUTES)
        .readTimeout(3, TimeUnit.MINUTES)
        .callTimeout(3, TimeUnit.MINUTES)
        .build()

    val api: OcrApi by lazy {
        Retrofit.Builder()
            .baseUrl(BuildConfig.API_BASE_URL)
            .client(httpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(OcrApi::class.java)
    }
}
