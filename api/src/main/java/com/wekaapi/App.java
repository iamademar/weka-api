package com.wekaapi;

import com.wekaapi.config.Config;
import com.wekaapi.controller.AlgorithmController;
import com.wekaapi.controller.DatasetController;
import com.wekaapi.controller.DiagnosticsController;
import com.wekaapi.controller.EdaController;
import com.wekaapi.controller.EvaluateController;
import com.wekaapi.controller.FilterController;
import com.wekaapi.controller.HealthController;
import com.wekaapi.controller.ModelController;
import com.wekaapi.controller.PredictController;
import com.wekaapi.controller.TrainController;
import com.wekaapi.controller.TransformController;
import com.wekaapi.error.ErrorHandler;
import com.wekaapi.service.DatasetService;
import com.wekaapi.service.DiagnosticsService;
import com.wekaapi.service.EdaService;
import com.wekaapi.service.FilterMetadataService;
import com.wekaapi.service.EvaluationService;
import com.wekaapi.service.ModelService;
import com.wekaapi.service.PredictionService;
import com.wekaapi.service.TrainingService;
import com.wekaapi.service.TransformService;
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
        EdaService edaService = new EdaService(datasetService);
        TransformService transformService = new TransformService(config, datasetService);
        DiagnosticsService diagnosticsService = new DiagnosticsService(evaluationService);
        FilterMetadataService filterMetadataService = new FilterMetadataService();

        HealthController health = new HealthController();
        AlgorithmController algorithms = new AlgorithmController();
        FilterController filters = new FilterController(filterMetadataService);
        DatasetController datasets = new DatasetController(datasetService);
        ModelController models = new ModelController(modelService);
        TrainController train = new TrainController(trainingService);
        PredictController predict = new PredictController(predictionService);
        EvaluateController evaluate = new EvaluateController(evaluationService);
        EdaController eda = new EdaController(edaService);
        TransformController transform = new TransformController(transformService);
        DiagnosticsController diagnostics = new DiagnosticsController(diagnosticsService);

        Javalin app = Javalin.create(cfg -> {
            cfg.showJavalinBanner = false;
            cfg.http.maxRequestSize = config.maxUploadBytes + (1024L * 1024L);
        });

        ErrorHandler.register(app);

        app.get("/health", health::get);
        app.get("/algorithms", algorithms::get);
        app.get("/filters", filters::get);
        app.get("/filters/metadata", filters::metadata);

        app.post("/datasets", datasets::upload);
        app.get("/datasets", datasets::list);
        app.get("/datasets/{name}", datasets::get);
        app.delete("/datasets/{name}", datasets::delete);

        app.get("/datasets/{name}/attribute-stats", eda::attributeStats);
        app.get("/datasets/{name}/summary", eda::summary);
        app.get("/datasets/{name}/histogram", eda::histogram);
        app.get("/datasets/{name}/scatter", eda::scatter);
        app.get("/datasets/{name}/scatter-matrix", eda::scatterMatrix);

        app.get("/models", models::list);
        app.get("/models/{name}", models::get);
        app.delete("/models/{name}", models::delete);
        app.get("/models/{name}/drawable-type", models::drawableType);
        app.get("/models/{name}/tree", models::tree);
        app.get("/models/{name}/graph", models::graph);

        app.post("/train", train::post);
        app.post("/predict", predict::post);
        app.post("/evaluate", evaluate::post);
        app.post("/transform", transform::post);
        app.post("/transform/preview", transform::preview);

        app.post("/diagnostics/errors", diagnostics::errors);
        app.post("/diagnostics/threshold-curve", diagnostics::thresholdCurve);
        app.post("/diagnostics/margin-curve", diagnostics::marginCurve);
        app.post("/diagnostics/cost-curve", diagnostics::costCurve);
        app.post("/diagnostics/calibration", diagnostics::calibration);

        return app;
    }
}
