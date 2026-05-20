package com.wekaapi;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.javalin.testtools.JavalinTest;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.InputStream;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class EvaluateIT {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir Path modelsDir;
    @TempDir Path dataDir;

    @Test
    public void evaluate_on_training_set_is_accurate() {
        JavalinTest.test(TestSupport.app(modelsDir, dataDir), (server, client) -> {
            byte[] iris;
            try (InputStream in = getClass().getResourceAsStream("/iris.arff")) {
                iris = in.readAllBytes();
            }

            MultipartBody body = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("name", "iris")
                    .addFormDataPart("file", "iris.arff",
                            RequestBody.create(iris, MediaType.parse("text/plain")))
                    .build();
            Request upload = new Request.Builder()
                    .url(client.getOrigin() + "/datasets")
                    .post(body)
                    .build();
            try (Response r = client.getOkHttp().newCall(upload).execute()) {
                assertEquals(201, r.code(), r.body().string());
            }

            String trainJson = "{\"dataset\":\"iris\",\"algorithm\":\"weka.classifiers.trees.J48\",\"modelName\":\"iris-j48\"}";
            Request train = new Request.Builder()
                    .url(client.getOrigin() + "/train")
                    .post(RequestBody.create(trainJson, MediaType.parse("application/json")))
                    .build();
            try (Response r = client.getOkHttp().newCall(train).execute()) {
                assertEquals(201, r.code(), r.body().string());
            }

            String evalJson = "{\"model\":\"iris-j48\",\"dataset\":\"iris\"}";
            Request eval = new Request.Builder()
                    .url(client.getOrigin() + "/evaluate")
                    .post(RequestBody.create(evalJson, MediaType.parse("application/json")))
                    .build();
            try (Response r = client.getOkHttp().newCall(eval).execute()) {
                String text = r.body().string();
                assertEquals(200, r.code(), text);
                JsonNode body2 = MAPPER.readTree(text);
                double accuracy = body2.get("accuracy").asDouble();
                assertTrue(accuracy > 0.9, "expected accuracy > 0.9, got " + accuracy);
                assertEquals(150, body2.get("numInstances").asInt());
            }
        });
    }
}
