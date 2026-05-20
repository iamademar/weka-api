package com.wekaapi.controller;

import com.wekaapi.dto.PredictRequest;
import com.wekaapi.error.ApiException;
import com.wekaapi.service.PredictionService;
import io.javalin.http.Context;

public class PredictController {

    private final PredictionService service;

    public PredictController(PredictionService service) {
        this.service = service;
    }

    public void post(Context ctx) {
        PredictRequest req;
        try {
            req = ctx.bodyAsClass(PredictRequest.class);
        } catch (Exception e) {
            throw new ApiException(400, "BAD_REQUEST", "invalid JSON: " + e.getMessage(), e);
        }
        ctx.json(service.predict(req));
    }
}
