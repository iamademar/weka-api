package com.wekaapi.controller;

import com.wekaapi.service.ModelService;
import io.javalin.http.Context;
import io.javalin.http.HttpStatus;

import java.util.Map;

public class ModelController {

    private final ModelService service;

    public ModelController(ModelService service) {
        this.service = service;
    }

    public void list(Context ctx) {
        ctx.json(Map.of("models", service.list()));
    }

    public void get(Context ctx) {
        String name = ctx.pathParam("name");
        ctx.json(service.describe(name));
    }

    public void delete(Context ctx) {
        String name = ctx.pathParam("name");
        service.delete(name);
        ctx.status(HttpStatus.NO_CONTENT);
    }
}
