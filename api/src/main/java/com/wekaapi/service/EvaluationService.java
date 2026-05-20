package com.wekaapi.service;

import com.wekaapi.dto.EvaluateRequest;
import com.wekaapi.error.ApiException;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class EvaluationService {

    private final DatasetService datasetService;
    private final ModelService modelService;

    public EvaluationService(DatasetService datasetService, ModelService modelService) {
        this.datasetService = datasetService;
        this.modelService = modelService;
    }

    public Map<String, Object> evaluate(EvaluateRequest req) {
        if (req == null || req.model == null || req.model.isBlank()) {
            throw new ApiException(400, "BAD_REQUEST", "model is required");
        }
        if (req.dataset == null || req.dataset.isBlank()) {
            throw new ApiException(400, "BAD_REQUEST", "dataset is required");
        }

        ModelService.LoadedModel loaded = modelService.load(req.model);
        Instances test = datasetService.load(req.dataset);
        test.setClassIndex(loaded.header().classIndex());

        Evaluation eval;
        try {
            eval = new Evaluation(loaded.header());
            eval.evaluateModel(loaded.classifier(), test);
        } catch (Exception e) {
            throw new ApiException(422, "EVALUATION_FAILED",
                    "evaluation failed: " + e.getMessage(), e);
        }

        Attribute classAttr = loaded.header().classAttribute();
        Map<String, Object> result = new java.util.LinkedHashMap<>();
        result.put("model", req.model);
        result.put("dataset", req.dataset);
        result.put("numInstances", (int) eval.numInstances());
        result.put("correct", (int) eval.correct());
        result.put("incorrect", (int) eval.incorrect());
        result.put("accuracy", round(eval.pctCorrect() / 100.0));

        if (classAttr.isNominal()) {
            result.put("kappa", round(eval.kappa()));
            result.put("weightedFMeasure", round(eval.weightedFMeasure()));
            try {
                double[][] matrix = eval.confusionMatrix();
                int[][] intMatrix = new int[matrix.length][];
                for (int i = 0; i < matrix.length; i++) {
                    intMatrix[i] = new int[matrix[i].length];
                    for (int j = 0; j < matrix[i].length; j++) {
                        intMatrix[i][j] = (int) Math.round(matrix[i][j]);
                    }
                }
                result.put("confusionMatrix", intMatrix);
            } catch (Exception ignored) {}
            List<String> labels = new ArrayList<>(classAttr.numValues());
            for (int i = 0; i < classAttr.numValues(); i++) labels.add(classAttr.value(i));
            result.put("classLabels", labels);
        }

        try {
            result.put("summary", eval.toSummaryString());
        } catch (Exception ignored) {}

        return result;
    }

    private static double round(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return 0.0;
        return Math.round(v * 10000.0) / 10000.0;
    }
}
