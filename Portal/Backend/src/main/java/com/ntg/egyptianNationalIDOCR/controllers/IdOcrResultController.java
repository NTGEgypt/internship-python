package com.ntg.egyptianNationalIDOCR.controllers;

import com.ntg.egyptianNationalIDOCR.dtos.ApiResponse;
import com.ntg.egyptianNationalIDOCR.dtos.IdOcrResultResponse;
import com.ntg.egyptianNationalIDOCR.dtos.ReviewRequest;
import com.ntg.egyptianNationalIDOCR.services.IdOcrResultService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/ocr-results")
public class IdOcrResultController {

    private final IdOcrResultService service;

    public IdOcrResultController(IdOcrResultService service) {
        this.service = service;
    }

    @GetMapping
    public ApiResponse getAllResults() {
        return service.getAllResults();
    }

    @GetMapping("/{id}")
    public ApiResponse getResultById(@PathVariable Long id) {
        return service.getResultById(id);
    }

    @GetMapping("/{id}/image")
    public ApiResponse getImage(@PathVariable Long id) {
        return service.getImage(id);
    }

    @PostMapping("/{id}/approve")
    public ApiResponse approve(
            @PathVariable Long id,
            @RequestBody ReviewRequest request
    ) {
        return service.approve(id, request);
    }

    @PostMapping("/{id}/reject")
    public ApiResponse reject(
            @PathVariable Long id,
            @RequestBody ReviewRequest request
    ) {
        return service.reject(id, request);
    }
}