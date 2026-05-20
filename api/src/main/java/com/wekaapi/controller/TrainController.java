package com.wekaapi.controller;

import com.wekaapi.dto.TrainRequest;
import com.wekaapi.error.ApiException;
import com.wekaapi.service.TrainingService;
import io.javalin.http.Context;
import io.javalin.http.HttpStatus;

public class TrainController {

    private final TrainingService service;

    public TrainController(TrainingService service) {
        this.service = service;
    }

    public void post(Context ctx) {
        TrainRequest req;
        try {
            req = ctx.bodyAsClass(TrainRequest.class);
        } catch (Exception e) {
            throw new ApiException(400, "BAD_REQUEST", "invalid JSON: " + e.getMessage(), e);
        }
        ctx.status(HttpStatus.CREATED).json(service.train(req));
    }
}
