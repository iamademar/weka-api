package com.wekaapi.service;

import com.wekaapi.config.Config;
import com.wekaapi.error.ApiException;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;

public class ModelService {

    public record LoadedModel(Classifier classifier, Instances header) {}

    private final Config config;
    private final Map<String, LoadedModel> cache = new ConcurrentHashMap<>();

    public ModelService(Config config) {
        this.config = config;
        try {
            Files.createDirectories(config.modelsDir);
        } catch (IOException e) {
            throw new RuntimeException("could not create models dir: " + config.modelsDir, e);
        }
    }

    public void save(String name, Classifier classifier, Instances header) {
        DatasetService.validateName(name);
        Path modelPath = modelPath(name);
        Path headerPath = headerPath(name);
        try {
            SerializationHelper.write(modelPath.toString(), classifier);
            Instances emptyHeader = new Instances(header, 0);
            emptyHeader.setClassIndex(header.classIndex());
            SerializationHelper.write(headerPath.toString(), emptyHeader);
        } catch (Exception e) {
            throw new ApiException(500, "INTERNAL_ERROR", "failed to persist model: " + e.getMessage(), e);
        }
        cache.put(name, new LoadedModel(classifier, header));
    }

    public LoadedModel load(String name) {
        DatasetService.validateName(name);
        LoadedModel cached = cache.get(name);
        if (cached != null) return cached;

        Path modelPath = modelPath(name);
        Path headerPath = headerPath(name);
        if (!Files.isRegularFile(modelPath)) {
            throw new ApiException(404, "MODEL_NOT_FOUND", "model not found: " + name);
        }
        try {
            Classifier classifier = (Classifier) SerializationHelper.read(modelPath.toString());
            Instances header;
            if (Files.isRegularFile(headerPath)) {
                header = (Instances) SerializationHelper.read(headerPath.toString());
            } else {
                throw new ApiException(500, "INTERNAL_ERROR", "model header missing for: " + name);
            }
            LoadedModel loaded = new LoadedModel(classifier, header);
            cache.put(name, loaded);
            return loaded;
        } catch (ApiException e) {
            throw e;
        } catch (Exception e) {
            throw new ApiException(500, "INTERNAL_ERROR", "failed to load model: " + e.getMessage(), e);
        }
    }

    public List<Map<String, Object>> list() {
        List<Map<String, Object>> out = new ArrayList<>();
        if (!Files.isDirectory(config.modelsDir)) return out;
        try (Stream<Path> stream = Files.list(config.modelsDir)) {
            stream.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().endsWith(".model"))
                    .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                    .forEach(p -> {
                        String fileName = p.getFileName().toString();
                        String name = fileName.substring(0, fileName.length() - ".model".length());
                        try {
                            out.add(Map.of("name", name, "sizeBytes", Files.size(p)));
                        } catch (IOException ignored) {}
                    });
        } catch (IOException e) {
            throw new ApiException(500, "INTERNAL_ERROR", e.getMessage(), e);
        }
        return out;
    }

    public Map<String, Object> describe(String name) {
        LoadedModel m = load(name);
        return Map.of(
                "name", name,
                "algorithm", m.classifier().getClass().getName(),
                "summary", m.classifier().toString()
        );
    }

    public void delete(String name) {
        DatasetService.validateName(name);
        Path modelPath = modelPath(name);
        Path headerPath = headerPath(name);
        if (!Files.isRegularFile(modelPath)) {
            throw new ApiException(404, "MODEL_NOT_FOUND", "model not found: " + name);
        }
        try {
            Files.deleteIfExists(modelPath);
            Files.deleteIfExists(headerPath);
        } catch (IOException e) {
            throw new ApiException(500, "INTERNAL_ERROR", "failed to delete model: " + e.getMessage(), e);
        }
        cache.remove(name);
    }

    private Path modelPath(String name) {
        return config.modelsDir.resolve(name + ".model");
    }

    private Path headerPath(String name) {
        return config.modelsDir.resolve(name + ".header");
    }
}
