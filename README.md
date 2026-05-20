# Weka REST API

HTTP API over [Weka](https://www.cs.waikato.ac.nz/ml/weka/) for training and serving classifiers — single-user local dev, no auth, persistence on the host filesystem.

- [Prerequisites](#prerequisites)
- [Local setup with Docker Compose](#local-setup-with-docker-compose)
- [Running the test suite](#running-the-test-suite)
- [Project layout](#project-layout)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [End-to-end walkthrough](#end-to-end-walkthrough)
- [Troubleshooting](#troubleshooting)
- [Security note](#security-note)

---

## Prerequisites

- **Docker Desktop** (or any Docker engine) with **Compose v2** (`docker compose ...`, not `docker-compose`).
- About 1 GB free disk for the first image build.
- Port `7070` free on `127.0.0.1`.

To run the JUnit test suite outside Docker you also need:

- **JDK 17** (`java -version` should report `17.x`).
- **Maven 3.8+** (`mvn -v`).

Install on macOS: `brew install --cask docker` and `brew install maven`.

---

## Local setup with Docker Compose

From the repo root (the directory containing this README):

```bash
docker compose up --build
```

What this does:

1. Builds the `api/` image via a multi-stage Dockerfile (Maven build on `eclipse-temurin:17-jdk`, runtime on `:17-jre`).
2. Produces a shaded uber jar at `/app/app.jar` inside the container.
3. Starts the container as `weka-api`, binds **only** `127.0.0.1:7070 → 7070`.
4. Mounts `./models` → `/app/models` and `./data` → `/app/data` so trained models and uploaded datasets persist across restarts.

The first build downloads Weka + Javalin + Jackson into the local Maven cache and takes a few minutes. Subsequent builds reuse the cache.

Verify it's up:

```bash
curl http://localhost:7070/health
# → {"status":"ok","wekaVersion":"3.9.6"}
```

### Common commands

```bash
# foreground, with logs
docker compose up --build

# background
docker compose up --build -d

# follow logs
docker compose logs -f weka-api

# stop (preserves models/ and data/)
docker compose down

# stop and remove built image (keep data)
docker compose down --rmi local

# rebuild from scratch
docker compose build --no-cache && docker compose up
```

### Persistence

`models/*.model`, `models/*.header`, and `data/*.{arff,csv}` live on the host. `docker compose down` does not delete them. To start clean:

```bash
docker compose down
rm -f models/* data/*  # keep the .gitkeep files
```

---

## Running the test suite

The suite uses Javalin's in-process test runner — no Docker, no network — and covers the 5 mandatory cases in `SPEC.md` §10.

```bash
cd api
mvn test
```

Expected output ends with something like `Tests run: 6, Failures: 0, Errors: 0, Skipped: 0`.

Tests:

- `HealthControllerTest` — `GET /health` returns 200 with a non-blank `wekaVersion`.
- `AlgorithmControllerTest` — `GET /algorithms` includes `weka.classifiers.trees.J48`.
- `TrainAndPredictIT` — upload iris → train J48 → predict 2 instances; distributions sum to ~1.0.
- `EvaluateIT` — train on iris, evaluate on iris, assert accuracy > 0.9.
- `SecurityTest` — `algorithm=java.lang.Runtime`, `modelName=../escape`, and `name=../foo` upload all return 400.

A sample `iris.arff` ships at `api/src/test/resources/iris.arff` and is reused by every test.

---

## Project layout

```
.
├── SPEC.md                # source-of-truth spec
├── README.md              # this file
├── compose.yaml           # Docker Compose service definition
├── api/                   # Javalin project
│   ├── Dockerfile         # multi-stage: build → runtime
│   ├── pom.xml            # Maven build, shade plugin → uber jar
│   └── src/
│       ├── main/java/com/wekaapi/
│       │   ├── App.java                # Javalin bootstrap + routes
│       │   ├── config/Config.java      # env-var loader
│       │   ├── controller/             # HTTP handlers
│       │   ├── service/                # business logic (Weka calls)
│       │   ├── dto/                    # request shapes
│       │   └── error/                  # ApiException + handler
│       └── test/                       # JUnit 5 tests + iris.arff
├── models/                # mounted into container, ignored by git
└── data/                  # mounted into container, ignored by git
```

---

## Configuration

All vars have defaults — `docker compose up` works without any `.env` file.

| Variable        | Default          | Purpose                                     |
| --------------- | ---------------- | ------------------------------------------- |
| `PORT`          | `7070`           | HTTP port Javalin binds to                  |
| `MODELS_DIR`    | `/app/models`    | Where serialized `.model` files live        |
| `DATA_DIR`      | `/app/data`      | Where uploaded ARFF/CSV files live          |
| `MAX_UPLOAD_MB` | `100`            | Reject dataset uploads above this size      |
| `LOG_LEVEL`     | `INFO`           | Root Logback level                          |

To override, edit the `environment:` block in `compose.yaml` or pass via `-e`:

```bash
docker compose run --rm -e LOG_LEVEL=DEBUG -p 127.0.0.1:7070:7070 weka-api
```

---

## API reference

Base URL: `http://localhost:7070`. All bodies are `application/json` unless stated. Errors return `{"error": "...", "code": "..."}` with an HTTP status from the table at the end of this section.

### `GET /health`

Liveness probe and Weka version.

```bash
curl http://localhost:7070/health
```

```json
{ "status": "ok", "wekaVersion": "3.9.6" }
```

---

### `GET /algorithms`

Lists Weka classifier classnames grouped by family (`trees`, `bayes`, `functions`, `lazy`, `rules`, `meta`, `misc`). Cached for the process lifetime.

```bash
curl http://localhost:7070/algorithms
```

```json
{
  "classifiers": {
    "bayes":     ["weka.classifiers.bayes.NaiveBayes", "..."],
    "functions": ["weka.classifiers.functions.Logistic", "..."],
    "lazy":      ["weka.classifiers.lazy.IBk", "..."],
    "meta":      ["weka.classifiers.meta.AdaBoostM1", "..."],
    "rules":     ["weka.classifiers.rules.JRip", "..."],
    "trees":     ["weka.classifiers.trees.J48", "weka.classifiers.trees.RandomForest", "..."]
  }
}
```

---

### Datasets

#### `POST /datasets` — upload a dataset

`multipart/form-data`:

| Field  | Required | Notes |
| ------ | -------- | ----- |
| `file` | yes      | An ARFF (`.arff`) or CSV (`.csv`) file. |
| `name` | no       | Stored filename base (no extension). Defaults to the uploaded filename minus its extension. Must not contain `/`, `\`, or `..`. |

```bash
curl -F file=@api/src/test/resources/iris.arff \
     -F name=iris \
     http://localhost:7070/datasets
```

`201 Created`:

```json
{
  "name": "iris",
  "path": "iris.arff",
  "format": "arff",
  "numInstances": 150,
  "numAttributes": 5,
  "classAttribute": "class"
}
```

The **last attribute is treated as the class** by convention.

#### `GET /datasets` — list

```bash
curl http://localhost:7070/datasets
```

```json
{ "datasets": [{ "name": "iris", "format": "arff", "sizeBytes": 7045 }] }
```

#### `GET /datasets/{name}` — metadata

```bash
curl http://localhost:7070/datasets/iris
```

```json
{
  "name": "iris",
  "format": "arff",
  "numInstances": 150,
  "attributes": [
    { "name": "sepallength", "type": "numeric" },
    { "name": "sepalwidth",  "type": "numeric" },
    { "name": "petallength", "type": "numeric" },
    { "name": "petalwidth",  "type": "numeric" },
    { "name": "class", "type": "nominal",
      "values": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"] }
  ],
  "classAttribute": "class"
}
```

#### `DELETE /datasets/{name}` — remove

```bash
curl -X DELETE http://localhost:7070/datasets/iris
# 204 No Content
```

---

### `POST /train` — train a classifier

Request body:

| Field        | Required | Notes |
| ------------ | -------- | ----- |
| `dataset`    | yes      | Name of an uploaded dataset. |
| `algorithm`  | yes      | Fully-qualified Weka classname; must start with `weka.classifiers.` (allowlist). |
| `options`    | no       | Weka CLI-style options as a string array, e.g. `["-C","0.25","-M","2"]`. |
| `modelName`  | yes      | Filename to persist to (no extension, no path separators). |
| `classIndex` | no       | Zero-based index. Default `-1` = last attribute. |

```bash
curl -X POST http://localhost:7070/train \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset":   "iris",
    "algorithm": "weka.classifiers.trees.J48",
    "options":   ["-C","0.25","-M","2"],
    "modelName": "iris-j48"
  }'
```

`201 Created`:

```json
{
  "modelName": "iris-j48",
  "algorithm": "weka.classifiers.trees.J48",
  "trainedOn": "iris",
  "trainingTimeMs": 142,
  "summary": "J48 pruned tree\n------------------\n..."
}
```

The classifier and dataset header are saved to `MODELS_DIR/iris-j48.model` and `MODELS_DIR/iris-j48.header`.

---

### Models

#### `GET /models` — list

```bash
curl http://localhost:7070/models
```

```json
{ "models": [{ "name": "iris-j48", "sizeBytes": 8123 }] }
```

#### `GET /models/{name}` — metadata

```bash
curl http://localhost:7070/models/iris-j48
```

```json
{
  "name": "iris-j48",
  "algorithm": "weka.classifiers.trees.J48",
  "summary": "J48 pruned tree\n------------------\n..."
}
```

#### `DELETE /models/{name}` — remove

Deletes the `.model` and `.header` files and invalidates the in-memory cache entry.

```bash
curl -X DELETE http://localhost:7070/models/iris-j48
# 204 No Content
```

---

### `POST /predict` — score new instances

| Field       | Required | Notes |
| ----------- | -------- | ----- |
| `model`     | yes      | Name of a trained model. |
| `instances` | yes      | Non-empty array. Each object maps attribute name → value. Missing keys are treated as missing values. |

```bash
curl -X POST http://localhost:7070/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "iris-j48",
    "instances": [
      {"sepallength":5.1,"sepalwidth":3.5,"petallength":1.4,"petalwidth":0.2},
      {"sepallength":6.7,"sepalwidth":3.0,"petallength":5.2,"petalwidth":2.3}
    ]
  }'
```

`200 OK` (nominal class):

```json
{
  "model": "iris-j48",
  "predictions": [
    { "predictedClass": "Iris-setosa",
      "distribution": { "Iris-setosa": 1.0, "Iris-versicolor": 0.0, "Iris-virginica": 0.0 } },
    { "predictedClass": "Iris-virginica",
      "distribution": { "Iris-setosa": 0.0, "Iris-versicolor": 0.02, "Iris-virginica": 0.98 } }
  ]
}
```

For a numeric class problem, `distribution` is omitted and `predictedClass` is the numeric value rendered as a string.

---

### `POST /evaluate` — score a model against a dataset

| Field     | Required | Notes |
| --------- | -------- | ----- |
| `model`   | yes      | Trained model name. |
| `dataset` | yes      | Test dataset name. Class index is taken from the model's saved header. |

```bash
curl -X POST http://localhost:7070/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-j48","dataset":"iris"}'
```

`200 OK`:

```json
{
  "model": "iris-j48",
  "dataset": "iris",
  "numInstances": 150,
  "correct": 147,
  "incorrect": 3,
  "accuracy": 0.98,
  "kappa": 0.97,
  "weightedFMeasure": 0.98,
  "confusionMatrix": [[50,0,0],[0,49,1],[0,2,48]],
  "classLabels": ["Iris-setosa","Iris-versicolor","Iris-virginica"],
  "summary": "Correctly Classified Instances ..."
}
```

---

### Error codes

| Code                  | HTTP | When                                                 |
| --------------------- | ---- | ---------------------------------------------------- |
| `DATASET_NOT_FOUND`   | 404  | No file at `DATA_DIR/{name}.{ext}`                   |
| `MODEL_NOT_FOUND`     | 404  | No file at `MODELS_DIR/{name}.model`                 |
| `INVALID_NAME`        | 400  | Name contains `/`, `\`, or `..`                      |
| `INVALID_ALGORITHM`   | 400  | Classname outside `weka.classifiers.` allowlist      |
| `INVALID_FORMAT`      | 400  | Unsupported file extension                           |
| `UPLOAD_TOO_LARGE`    | 413  | Exceeds `MAX_UPLOAD_MB`                              |
| `TRAINING_FAILED`     | 422  | Weka threw during `buildClassifier`                  |
| `PREDICTION_FAILED`   | 422  | Weka threw during prediction                         |
| `EVALUATION_FAILED`   | 422  | Weka threw during evaluation                         |
| `BAD_REQUEST`         | 400  | Malformed JSON or missing required fields            |
| `INTERNAL_ERROR`      | 500  | Anything uncaught (also logged with full stacktrace) |

---

## End-to-end walkthrough

Once the stack is up:

```bash
# 1. upload the sample iris dataset that ships with the repo
curl -F file=@api/src/test/resources/iris.arff \
     -F name=iris \
     http://localhost:7070/datasets

# 2. train a J48 decision tree
curl -X POST http://localhost:7070/train \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"iris","algorithm":"weka.classifiers.trees.J48","modelName":"iris-j48"}'

# 3. predict on two flowers
curl -X POST http://localhost:7070/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-j48","instances":[
        {"sepallength":5.1,"sepalwidth":3.5,"petallength":1.4,"petalwidth":0.2},
        {"sepallength":6.7,"sepalwidth":3.0,"petallength":5.2,"petalwidth":2.3}
      ]}'

# 4. evaluate against the training set (sanity check)
curl -X POST http://localhost:7070/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-j48","dataset":"iris"}'

# 5. confirm persistence: stop and restart
docker compose down
docker compose up -d
curl http://localhost:7070/models   # iris-j48 still listed
```

---

## Troubleshooting

**`docker: command not found` / Docker isn't running.**
Install/start Docker Desktop. `docker info` should return a daemon block.

**`docker compose` complains it isn't a recognised command.**
You're on Compose v1. Either upgrade to Compose v2 (ships with Docker Desktop) or substitute `docker-compose` everywhere in this README.

**`bind: address already in use` on port 7070.**
Something already owns that port. Either kill it (`lsof -nP -iTCP:7070 -sTCP:LISTEN`) or change the host port mapping in `compose.yaml`:

```yaml
ports:
  - "127.0.0.1:7171:7070"   # external 7171 → internal 7070
```

**First build hangs at `dependency:go-offline`.**
Maven is fetching Weka (~20 MB) and its transitive deps. Give it 2–5 minutes on a fresh cache. Subsequent builds reuse the BuildKit cache mount.

**Build fails resolving `weka-dev:3.9.7-SNAPSHOT`.**
The repo defaults to the stable `3.9.6` release for exactly this reason. If you want to try the snapshot, set `<weka.version>3.9.7-SNAPSHOT</weka.version>` in `api/pom.xml` — it's pulled from `https://oss.sonatype.org/content/repositories/snapshots/`.

**`GET /algorithms` returns a short list.**
The endpoint relies on Weka's `ClassDiscovery`. If it can't enumerate the classpath at runtime, the controller falls back to a curated set of common classifiers (J48, RandomForest, NaiveBayes, Logistic, IBk, etc.). All Weka classifiers on the classpath are still usable via `/train` — the listing is for convenience.

**`/predict` returns `400 BAD_REQUEST` "attribute X value not in domain".**
Nominal attributes only accept values from the set declared in the training data. Check `GET /datasets/{name}` to see the allowed values.

**`/predict` returns `404 MODEL_NOT_FOUND` after restart.**
Make sure `./models` is bind-mounted (it is by default in `compose.yaml`). If you started the container with `docker run` directly without the mount, models won't persist.

**`mvn test` fails with `class file has wrong version`.**
You're on a JDK older than 17. Install JDK 17 (`brew install --cask temurin@17`) and either set `JAVA_HOME` or use `jenv`/`sdkman` to switch.

**Container starts but `curl` hangs.**
Check the logs: `docker compose logs weka-api`. If you see `Address already in use`, see the port note above. If the JVM crashed during init, the stack trace will be there.

---

## Weka version

Builds against the stable `weka-dev` release (`3.9.6`) from Maven Central. The spec preferred `3.9.7-SNAPSHOT` but allows the documented downgrade when the snapshot is unavailable; flip `<weka.version>` in `api/pom.xml` if you want the snapshot.
