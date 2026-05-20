package com.wekaapi.service;

import com.wekaapi.dto.TrainRequest;
import com.wekaapi.error.ApiException;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.Map;

public class TrainingService {

    private static final String ALLOWED_PREFIX = "weka.classifiers.";

    private final DatasetService datasetService;
    private final ModelService modelService;

    public TrainingService(DatasetService datasetService, ModelService modelService) {
        this.datasetService = datasetService;
        this.modelService = modelService;
    }

    public Map<String, Object> train(TrainRequest req) {
        if (req == null) throw new ApiException(400, "BAD_REQUEST", "missing request body");
        if (req.dataset == null || req.dataset.isBlank())
            throw new ApiException(400, "BAD_REQUEST", "dataset is required");
        if (req.algorithm == null || req.algorithm.isBlank())
            throw new ApiException(400, "BAD_REQUEST", "algorithm is required");
        if (req.modelName == null || req.modelName.isBlank())
            throw new ApiException(400, "BAD_REQUEST", "modelName is required");

        DatasetService.validateName(req.modelName);

        if (!req.algorithm.startsWith(ALLOWED_PREFIX)) {
            throw new ApiException(400, "INVALID_ALGORITHM",
                    "algorithm must start with '" + ALLOWED_PREFIX + "': " + req.algorithm);
        }

        Class<?> raw;
        try {
            raw = Class.forName(req.algorithm);
        } catch (ClassNotFoundException e) {
            throw new ApiException(400, "INVALID_ALGORITHM", "unknown algorithm: " + req.algorithm);
        }
        if (!Classifier.class.isAssignableFrom(raw)) {
            throw new ApiException(400, "INVALID_ALGORITHM",
                    "algorithm is not a Weka Classifier: " + req.algorithm);
        }

        String[] options = (req.options == null) ? new String[0] : req.options.toArray(new String[0]);

        Classifier classifier;
        try {
            classifier = AbstractClassifier.forName(req.algorithm, options);
        } catch (Exception e) {
            throw new ApiException(400, "INVALID_ALGORITHM",
                    "failed to instantiate classifier: " + e.getMessage(), e);
        }

        Instances data = datasetService.load(req.dataset);
        int classIndex = (req.classIndex == null || req.classIndex == -1)
                ? data.numAttributes() - 1
                : req.classIndex;
        if (classIndex < 0 || classIndex >= data.numAttributes()) {
            throw new ApiException(400, "BAD_REQUEST", "classIndex out of range: " + classIndex);
        }
        data.setClassIndex(classIndex);

        long start = System.currentTimeMillis();
        try {
            classifier.buildClassifier(data);
        } catch (Exception e) {
            throw new ApiException(422, "TRAINING_FAILED",
                    "training failed: " + e.getMessage(), e);
        }
        long elapsed = System.currentTimeMillis() - start;

        modelService.save(req.modelName, classifier, data);

        return Map.of(
                "modelName", req.modelName,
                "algorithm", req.algorithm,
                "trainedOn", req.dataset,
                "trainingTimeMs", elapsed,
                "summary", classifier.toString()
        );
    }
}
