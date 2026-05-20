package com.wekaapi.service;

import com.wekaapi.dto.EvaluateRequest;
import com.wekaapi.error.ApiException;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class EvaluationService {

    public record EvaluationResult(Evaluation evaluation, ModelService.LoadedModel loaded, Instances test) {}

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
        EvaluationResult result = runEvaluation(req.model, req.dataset);
        return summarize(result, req.model, req.dataset);
    }

    public EvaluationResult runEvaluation(String modelName, String datasetName) {
        ModelService.LoadedModel loaded = modelService.load(modelName);
        Instances test = datasetService.load(datasetName);
        test.setClassIndex(loaded.header().classIndex());
        try {
            Evaluation eval = new Evaluation(loaded.header());
            eval.evaluateModel(loaded.classifier(), test);
            return new EvaluationResult(eval, loaded, test);
        } catch (Exception e) {
            throw new ApiException(422, "EVALUATION_FAILED",
                    "evaluation failed: " + e.getMessage(), e);
        }
    }

    public Map<String, Object> summarize(EvaluationResult r, String modelName, String datasetName) {
        Evaluation eval = r.evaluation();
        Attribute classAttr = r.loaded().header().classAttribute();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("model", modelName);
        out.put("dataset", datasetName);
        out.put("numInstances", (int) eval.numInstances());
        out.put("correct", (int) eval.correct());
        out.put("incorrect", (int) eval.incorrect());
        out.put("accuracy", round(eval.pctCorrect() / 100.0));

        if (classAttr.isNominal()) {
            out.put("kappa", round(eval.kappa()));
            out.put("weightedFMeasure", round(eval.weightedFMeasure()));
            try {
                double[][] matrix = eval.confusionMatrix();
                int[][] intMatrix = new int[matrix.length][];
                for (int i = 0; i < matrix.length; i++) {
                    intMatrix[i] = new int[matrix[i].length];
                    for (int j = 0; j < matrix[i].length; j++) {
                        intMatrix[i][j] = (int) Math.round(matrix[i][j]);
                    }
                }
                out.put("confusionMatrix", intMatrix);
            } catch (Exception ignored) {}
            List<String> labels = new ArrayList<>(classAttr.numValues());
            for (int i = 0; i < classAttr.numValues(); i++) labels.add(classAttr.value(i));
            out.put("classLabels", labels);
        }

        try {
            out.put("summary", eval.toSummaryString());
        } catch (Exception ignored) {}

        return out;
    }

    private static double round(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return 0.0;
        return Math.round(v * 10000.0) / 10000.0;
    }
}
