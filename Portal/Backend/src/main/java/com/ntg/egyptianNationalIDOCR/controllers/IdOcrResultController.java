package com.ntg.egyptianNationalIDOCR.controllers;

import com.ntg.egyptianNationalIDOCR.dtos.IdOcrResultResponse;
import com.ntg.egyptianNationalIDOCR.dtos.ReviewRequest;
import com.ntg.egyptianNationalIDOCR.services.IdOcrResultService;
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
    public List<IdOcrResultResponse> getAllResults() {
        return service.getAllResults();
    }

    @PostMapping("/{id}/approve")
    public IdOcrResultResponse approve(
            @PathVariable Long id,
            @RequestBody ReviewRequest request
    ) {
        return service.approve(id, request);
    }
}