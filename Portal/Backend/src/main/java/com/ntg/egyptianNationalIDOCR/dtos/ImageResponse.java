package com.ntg.egyptianNationalIDOCR.dtos;

public class ImageResponse {

    private String mimeType;
    private String image;

    public ImageResponse(String mimeType, String image) {
        this.mimeType = mimeType;
        this.image = image;
    }

    public String getMimeType() {
        return mimeType;
    }

    public String getImage() {
        return image;
    }
}