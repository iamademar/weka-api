package com.wekaapi.error;

import io.javalin.Javalin;
import io.javalin.http.HttpStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

public final class ErrorHandler {

    private static final Logger LOG = LoggerFactory.getLogger(ErrorHandler.class);

    private ErrorHandler() {}

    public static void register(Javalin app) {
        app.exception(ApiException.class, (ex, ctx) -> {
            ctx.status(ex.status());
            ctx.json(Map.of("error", ex.getMessage(), "code", ex.code()));
        });

        app.exception(IllegalArgumentException.class, (ex, ctx) -> {
            ctx.status(HttpStatus.BAD_REQUEST);
            ctx.json(Map.of("error", ex.getMessage(), "code", "BAD_REQUEST"));
        });

        app.exception(Exception.class, (ex, ctx) -> {
            LOG.error("Unhandled exception", ex);
            ctx.status(HttpStatus.INTERNAL_SERVER_ERROR);
            ctx.json(Map.of("error", ex.getMessage() == null ? "internal error" : ex.getMessage(),
                            "code", "INTERNAL_ERROR"));
        });
    }
}
