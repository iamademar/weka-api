package com.wekaapi.service;

import com.wekaapi.dto.FilterSpec;
import com.wekaapi.dto.TrainRequest;
import com.wekaapi.error.ApiException;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.MultiFilter;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
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

        Classifier baseClassifier;
        try {
            baseClassifier = AbstractClassifier.forName(req.algorithm, options);
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

        Classifier classifier = baseClassifier;
        List<String> appliedFilters = new ArrayList<>();
        if (req.filters != null && !req.filters.isEmpty()) {
            classifier = wrapWithFilters(baseClassifier, req.filters, appliedFilters);
        }

        long start = System.currentTimeMillis();
        try {
            classifier.buildClassifier(data);
        } catch (Exception e) {
            throw new ApiException(422, "TRAINING_FAILED",
                    "training failed: " + e.getMessage(), e);
        }
        long elapsed = System.currentTimeMillis() - start;

        modelService.save(req.modelName, classifier, data);

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("modelName", req.modelName);
        response.put("algorithm", req.algorithm);
        response.put("trainedOn", req.dataset);
        response.put("trainingTimeMs", elapsed);
        response.put("summary", classifier.toString());
        if (!appliedFilters.isEmpty()) {
            response.put("filters", appliedFilters);
        }
        return response;
    }

    private static Classifier wrapWithFilters(Classifier base, List<FilterSpec> specs, List<String> applied) {
        Filter filter;
        if (specs.size() == 1) {
            filter = TransformService.buildFilter(specs.get(0));
            applied.add(specs.get(0).filter);
        } else {
            Filter[] chain = new Filter[specs.size()];
            for (int i = 0; i < specs.size(); i++) {
                chain[i] = TransformService.buildFilter(specs.get(i));
                applied.add(specs.get(i).filter);
            }
            MultiFilter mf = new MultiFilter();
            mf.setFilters(chain);
            filter = mf;
        }
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(filter);
        fc.setClassifier(base);
        return fc;
    }
}
