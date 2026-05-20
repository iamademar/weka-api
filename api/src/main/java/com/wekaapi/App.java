package com.wekaapi;

import com.wekaapi.config.Config;
import com.wekaapi.controller.AlgorithmController;
import com.wekaapi.controller.DatasetController;
import com.wekaapi.controller.EvaluateController;
import com.wekaapi.controller.HealthController;
import com.wekaapi.controller.ModelController;
import com.wekaapi.controller.PredictController;
import com.wekaapi.controller.TrainController;
import com.wekaapi.error.ErrorHandler;
import com.wekaapi.service.DatasetService;
import com.wekaapi.service.EvaluationService;
import com.wekaapi.service.ModelService;
import com.wekaapi.service.PredictionService;
import com.wekaapi.service.TrainingService;
import io.javalin.Javalin;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class App {

    private static final Logger LOG = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) {
        Config config = Config.fromEnv();
        Javalin app = build(config);
        app.start(config.port);
        LOG.info("Weka API listening on port {}", config.port);
    }

    public static Javalin build(Config config) {
        DatasetService datasetService = new DatasetService(config);
        ModelService modelService = new ModelService(config);
        TrainingService trainingService = new TrainingService(datasetService, modelService);
        PredictionService predictionService = new PredictionService(modelService);
        EvaluationService evaluationService = new EvaluationService(datasetService, modelService);

        HealthController health = new HealthController();
        AlgorithmController algorithms = new AlgorithmController();
        DatasetController datasets = new DatasetController(datasetService);
        ModelController models = new ModelController(modelService);
        TrainController train = new TrainController(trainingService);
        PredictController predict = new PredictController(predictionService);
        EvaluateController evaluate = new EvaluateController(evaluationService);

        Javalin app = Javalin.create(cfg -> {
            cfg.showJavalinBanner = false;
            cfg.http.maxRequestSize = config.maxUploadBytes + (1024L * 1024L);
        });

        ErrorHandler.register(app);

        app.get("/health", health::get);
        app.get("/algorithms", algorithms::get);

        app.post("/datasets", datasets::upload);
        app.get("/datasets", datasets::list);
        app.get("/datasets/{name}", datasets::get);
        app.delete("/datasets/{name}", datasets::delete);

        app.get("/models", models::list);
        app.get("/models/{name}", models::get);
        app.delete("/models/{name}", models::delete);

        app.post("/train", train::post);
        app.post("/predict", predict::post);
        app.post("/evaluate", evaluate::post);

        return app;
    }
}
