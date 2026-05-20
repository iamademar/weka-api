# Weka REST API

HTTP API over [Weka](https://www.cs.waikato.ac.nz/ml/weka/) for training and serving classifiers — single-user local dev, no auth, persistence on the host filesystem.

- [Prerequisites](#prerequisites)
- [Local setup with Docker Compose](#local-setup-with-docker-compose)
- [Running the test suite](#running-the-test-suite)
- [Project layout](#project-layout)
- [Configuration](#configuration)
- [API reference](#api-reference)
  - [EDA / data exploration](#eda--data-exploration)
  - [Filters and transform](#filters-and-transform)
  - [Model structure](#model-structure)
  - [Post-training diagnostics](#post-training-diagnostics)
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

The suite uses Javalin's in-process test runner — no running container required — and currently includes **22 tests** covering health, algorithms, security, the train → predict → evaluate flow, EDA, transform (apply + preview), filter metadata, leakage-safe filtered training, model-structure (tree/graph), and the five diagnostics endpoints.

### Locally (needs JDK 17 + Maven on your machine)

```bash
cd api
mvn test
```

Expected output ends with `Tests run: 22, Failures: 0, Errors: 0, Skipped: 0`.

### Via Docker (no local JDK/Maven needed)

If you'd rather not install Java locally, run Maven inside a throwaway container against your source tree:

```bash
cd /path/to/repo
docker run --rm \
  -v "$(pwd)/api":/build \
  -v ~/.m2:/root/.m2 \
  -w /build \
  maven:3.9-eclipse-temurin-17 \
  mvn -B test
```

- `--rm` deletes the container when it exits
- `-v "$(pwd)/api":/build` mounts the source
- `-v ~/.m2:/root/.m2` shares your Maven cache so re-runs are fast (~2s after warm-up)
- `maven:3.9-eclipse-temurin-17` is the official Maven image bundled with JDK 17

Target a single class with `-Dtest`:

```bash
docker run --rm -v "$(pwd)/api":/build -v ~/.m2:/root/.m2 -w /build \
  maven:3.9-eclipse-temurin-17 \
  mvn -B test -Dtest=DiagnosticsIT
```

### Optional: bake a `test` service into `compose.yaml`

If you'd like `docker compose run test` to be the canonical command, append a service to `compose.yaml`:

```yaml
  test:
    image: maven:3.9-eclipse-temurin-17
    working_dir: /build
    volumes:
      - ./api:/build
      - maven-cache:/root/.m2
    command: ["mvn", "-B", "test"]
    profiles: ["test"]

volumes:
  maven-cache:
```

`profiles: ["test"]` keeps the service out of `docker compose up`. Run it explicitly:

```bash
docker compose run --rm test                            # all tests
docker compose run --rm test mvn -B test -Dtest=EdaIT   # one class
```

### Test inventory

- `HealthControllerTest` — `GET /health` returns 200 with a non-blank `wekaVersion`.
- `AlgorithmControllerTest` — `GET /algorithms` includes `weka.classifiers.trees.J48`.
- `SecurityTest` — `algorithm=java.lang.Runtime`, `modelName=../escape`, and `name=../foo` upload all return 400.
- `TrainAndPredictIT` — upload iris → train J48 → predict 2 instances; distributions sum to ~1.0.
- `EvaluateIT` — train on iris, evaluate on iris, assert accuracy > 0.9.
- `EdaIT` — exercises all 5 EDA endpoints + `INVALID_ATTRIBUTE` guard.
- `TransformIT` — `/filters` listing, Normalize chain, and `INVALID_FILTER` guard.
- `TransformPreviewIT` — preview returns sample rows, caps head at 200, never writes to `DATA_DIR`.
- `FilterMetadataIT` — listing exposes `supervised`/`level` flags; per-filter metadata returns description + options schema; allowlist + missing-param errors.
- `TrainFilteredIT` — train + predict + evaluate + Drawable extraction through `FilteredClassifier`; rejects non-Weka filters; multi-filter chains echoed in response.
- `ModelGraphIT` — J48 returns DOT tree, Logistic returns `NOT_DRAWABLE`.
- `DiagnosticsIT` — exercises all 5 diagnostics endpoints + `INVALID_CLASS_VALUE` guard.

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

### EDA / data exploration

All five EDA endpoints take a dataset name in the path. None of them mutate state. They share a common sampling convention: pass `sample` (max 5000, default 500) and `seed` (default 42) to control the random shuffle used for scatter-style endpoints.

#### `GET /datasets/{name}/attribute-stats?attribute=X`

Per-attribute summary using Weka's `AttributeStats`.

```bash
curl 'http://localhost:7070/datasets/iris/attribute-stats?attribute=petallength'
```

Numeric:

```json
{
  "name": "petallength",
  "type": "numeric",
  "count": 150, "missing": 0, "distinct": 43, "unique": 2,
  "numeric": { "min": 1.0, "max": 6.9, "mean": 3.7587, "stdDev": 1.7644, "sum": 563.8 }
}
```

Nominal:

```json
{
  "name": "class",
  "type": "nominal",
  "count": 150, "missing": 0, "distinct": 3, "unique": 0,
  "nominalCounts": { "Iris-setosa": 50, "Iris-versicolor": 50, "Iris-virginica": 50 }
}
```

#### `GET /datasets/{name}/summary`

Bulk version — returns the stats above for every attribute on the dataset.

```bash
curl http://localhost:7070/datasets/iris/summary
```

#### `GET /datasets/{name}/histogram?attribute=X&bins=10&groupBy=class`

For numeric attributes, equal-width bins between `min` and `max`. For nominal, one bin per value. Pass `groupBy=class` to break each bin down by class label.

```bash
curl 'http://localhost:7070/datasets/iris/histogram?attribute=petallength&bins=5&groupBy=class'
```

```json
{
  "attribute": "petallength",
  "type": "numeric",
  "bins": [
    { "lo": 1.0, "hi": 2.18, "count": 50, "byClass": { "Iris-setosa": 50, "Iris-versicolor": 0, "Iris-virginica": 0 } },
    { "lo": 2.18, "hi": 3.36, "count": 0,  "byClass": { "Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0 } }
  ],
  "missing": 0,
  "classLabels": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
}
```

#### `GET /datasets/{name}/scatter?x=A&y=B&sample=500&jitter=false&seed=42`

Per-point JSON for an x/y plot. Points carry a `class` field when the dataset has a nominal class. Set `jitter=true` to add Gaussian noise when an axis is nominal (does not mutate the dataset).

```bash
curl 'http://localhost:7070/datasets/iris/scatter?x=petallength&y=petalwidth&sample=200'
```

```json
{
  "x": "petallength", "y": "petalwidth",
  "xType": "numeric", "yType": "numeric",
  "classAttribute": "class",
  "totalInstances": 150, "sampled": 150,
  "points": [{ "x": 1.4, "y": 0.2, "class": "Iris-setosa" }]
}
```

#### `GET /datasets/{name}/scatter-matrix?attributes=A,B,C&sample=500`

All unordered pairs of the listed attributes (server caps at 6 attributes → up to 15 pairs).

```bash
curl 'http://localhost:7070/datasets/iris/scatter-matrix?attributes=sepallength,sepalwidth,petallength&sample=200'
```

```json
{
  "attributes": ["sepallength", "sepalwidth", "petallength"],
  "classAttribute": "class",
  "totalInstances": 150, "sampled": 150,
  "pairs": [
    { "x": "sepallength", "y": "sepalwidth", "points": [{ "x": 5.1, "y": 3.5, "class": "Iris-setosa" }] }
  ]
}
```

---

### Filters and transform

#### `GET /filters`

Lists every Weka filter discoverable on the classpath, grouped by family (`unsupervised.attribute`, `unsupervised.instance`, `supervised.attribute`, `supervised.instance`, plus `misc` for top-level filters like `MultiFilter`). Each entry carries flags so the client picker can distinguish leakage-prone supervised filters from safe unsupervised ones.

```bash
curl http://localhost:7070/filters
```

```json
{
  "filters": {
    "unsupervised.attribute": [
      { "classname": "weka.filters.unsupervised.attribute.Normalize",
        "supervised": false,
        "level": "attribute" }
    ],
    "supervised.attribute": [
      { "classname": "weka.filters.supervised.attribute.Discretize",
        "supervised": true,
        "level": "attribute" }
    ],
    "misc": [
      { "classname": "weka.filters.AllFilter",
        "supervised": null,
        "level": null }
    ]
  }
}
```

`supervised` is derived from the `SupervisedFilter` / `UnsupervisedFilter` marker interfaces — `null` for top-level filters that implement neither.

#### `GET /filters/metadata?filter=<fqn>`

Per-filter introspection — `globalInfo()` description and the full `listOptions()` schema, so a client picker can render an options form without hardcoding any filter knowledge.

```bash
curl 'http://localhost:7070/filters/metadata?filter=weka.filters.unsupervised.attribute.Normalize'
```

```json
{
  "classname": "weka.filters.unsupervised.attribute.Normalize",
  "supervised": false,
  "level": "attribute",
  "family": "unsupervised.attribute",
  "description": "Normalizes all numeric values in the given dataset (apart from the class attribute, if set)...",
  "options": [
    { "name": "S", "synopsis": "-S <num>", "description": "The scaling factor (default 1.0).", "numArguments": 1, "default": "1.0" },
    { "name": "T", "synopsis": "-T <num>", "description": "The translation (default 0.0).", "numArguments": 1, "default": "0.0" },
    { "name": "unset-class-temporarily", "synopsis": "-unset-class-temporarily", "description": "...", "numArguments": 0, "default": false }
  ]
}
```

For boolean flags (`numArguments == 0`), `default` is the literal `true`/`false` — Weka boolean flags default to off when absent. For value flags (`numArguments >= 1`), `default` is the stringified value (omitted when no default is set).

`400 INVALID_FILTER` if the FQN is outside the `weka.filters.` allowlist or doesn't resolve to a `Filter`. `400 BAD_REQUEST` if the `filter` query param is missing.

#### `POST /transform`

Apply a chain of filters to an existing dataset and persist the result as a new dataset.

| Field        | Required | Notes |
| ------------ | -------- | ----- |
| `dataset`    | yes      | Source dataset name. |
| `filters`    | yes      | Non-empty array. Each element: `{"filter": "<weka.filters.* FQN>", "options": [...]}`. Filter must start with `weka.filters.` (allowlist). |
| `outputName` | yes      | New dataset name (no extension, no path separators). |
| `format`     | no       | `"arff"` (default) or `"csv"`. |

```bash
curl -X POST http://localhost:7070/transform \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": "iris",
    "filters": [
      {"filter": "weka.filters.unsupervised.attribute.Normalize", "options": []}
    ],
    "outputName": "iris-norm"
  }'
```

`201 Created`:

```json
{
  "name": "iris-norm",
  "format": "arff",
  "path": "iris-norm.arff",
  "numInstances": 150,
  "numAttributes": 5,
  "filtersApplied": ["weka.filters.unsupervised.attribute.Normalize"]
}
```

The new dataset is immediately usable for `/train`, `/evaluate`, EDA endpoints, etc.

**Note on PCA:** `weka.filters.unsupervised.attribute.PrincipalComponents` works on datasets with a nominal class — it ignores the class attribute when computing components and passes it through unchanged.

#### `POST /transform/preview`

Same request body as `POST /transform` minus `outputName` (ignored if present). Runs the chain **in memory** and returns metadata plus a small sample of transformed rows. Nothing is written to `DATA_DIR` — use this to iterate on a filter chain before committing.

| Query | Default | Bounds |
| --- | --- | --- |
| `head` | 20 | 1 to 200 |
| `seed` | 42 | any long (controls the row shuffle) |

```bash
curl -X POST 'http://localhost:7070/transform/preview?head=10' \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": "iris",
    "filters": [
      {"filter": "weka.filters.unsupervised.attribute.Normalize", "options": []}
    ]
  }'
```

```json
{
  "dataset": "iris",
  "numInstances": 150,
  "totalInstances": 150,
  "sampled": 10,
  "numAttributes": 5,
  "attributes": [
    { "name": "petallength", "type": "numeric" },
    { "name": "class", "type": "nominal", "values": ["Iris-setosa","Iris-versicolor","Iris-virginica"] }
  ],
  "filtersApplied": ["weka.filters.unsupervised.attribute.Normalize"],
  "head": [
    { "sepallength": 0.222, "sepalwidth": 0.625, "petallength": 0.068, "petalwidth": 0.042, "class": "Iris-setosa" }
  ]
}
```

---

### Preprocess → train workflow

There are two ways to apply preprocessing before training, and **they're not equivalent**.

#### Path A: `POST /transform` → `POST /train` on the new dataset

Use when the chain is purely **unsupervised** (`Normalize`, `Standardize`, `Discretize` unsupervised, `PrincipalComponents`, `ReplaceMissingValues`, etc.). The filter is fit on the full dataset once, the result is persisted, and you train on it like any other dataset.

```bash
# 1. upload raw data
curl -F file=@iris.arff -F name=iris http://localhost:7070/datasets

# 2. iterate on the filter chain in memory
curl -X POST 'http://localhost:7070/transform/preview?head=5' \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"iris","filters":[{"filter":"weka.filters.unsupervised.attribute.Normalize","options":[]}]}'

# 3. happy with the chain → commit it
curl -X POST http://localhost:7070/transform \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"iris","filters":[{"filter":"weka.filters.unsupervised.attribute.Normalize","options":[]}],"outputName":"iris-norm"}'

# 4. train on the new dataset
curl -X POST http://localhost:7070/train \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"iris-norm","algorithm":"weka.classifiers.trees.J48","modelName":"iris-norm-j48"}'

# 5. predict/evaluate — features sent in are the NORMALIZED ones
curl -X POST http://localhost:7070/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-norm-j48","dataset":"iris-norm"}'
```

#### ⚠️ Leakage warning for supervised filters

Path A is **unsafe** when the filter is supervised — i.e. when it looks at the class attribute to decide its parameters. Common offenders:

- `weka.filters.supervised.attribute.Discretize` — Fayyad–Irani MDL discretization uses class info
- `weka.filters.supervised.attribute.AttributeSelection` — feature selection by class-driven scoring
- `weka.filters.supervised.attribute.NominalToBinary` — class-aware one-hot
- `weka.filters.supervised.attribute.MergeNominalValues`

Running these via `/transform` on your full dataset and then training on the result leaks class signal into the features. Cross-validation on the transformed dataset will report optimistic accuracy that won't hold on held-out data. The `supervised: true` flag in `/filters` is your signal to use Path B instead.

#### Path B: `POST /train` with embedded filter chain (leakage-safe)

Use whenever any filter in your chain is supervised. The request looks the same as a normal `/train`, but with an extra `filters` array. The API wraps your classifier in `weka.classifiers.meta.FilteredClassifier`; the filter is fit on each training fold's data only, not the whole dataset.

```bash
# 1. train — filters embedded in the model, raw dataset name
curl -X POST http://localhost:7070/train \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": "iris",
    "algorithm": "weka.classifiers.trees.J48",
    "modelName": "iris-fc-j48",
    "filters": [
      {"filter": "weka.filters.supervised.attribute.AttributeSelection",
       "options": ["-E","weka.attributeSelection.CfsSubsetEval","-S","weka.attributeSelection.BestFirst"]}
    ]
  }'

# 2. predict on RAW features — the model applies the filter chain internally
curl -X POST http://localhost:7070/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-fc-j48","instances":[{"sepallength":5.1,"sepalwidth":3.5,"petallength":1.4,"petalwidth":0.2}]}'

# 3. evaluate on the RAW dataset — same story
curl -X POST http://localhost:7070/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-fc-j48","dataset":"iris"}'
```

The training response echoes the applied filter chain in a `filters` field. All downstream endpoints — `/predict`, `/evaluate`, `/diagnostics/*`, `/models/{name}/tree` — work transparently; the wrapped classifier delegates `Drawable`/`Classifier` calls through to the base learner.

Single-filter chains skip `MultiFilter` for efficiency; chains of two or more get bundled into a `weka.filters.MultiFilter` automatically.

---

### Model structure

For classifiers that implement `weka.core.Drawable` (most tree classifiers and Bayes nets), you can extract the model structure as Graphviz DOT.

#### `GET /models/{name}/drawable-type`

```bash
curl http://localhost:7070/models/iris-j48/drawable-type
```

```json
{ "name": "iris-j48", "type": "tree" }
```

Possible types: `"tree"`, `"graph"` (Bayes net), `"newick"`, or `"none"`.

#### `GET /models/{name}/tree`

Returns the classifier's tree in Graphviz DOT for tree-based classifiers (J48, RandomTree, REPTree, M5P, LMT, HoeffdingTree). 400 `NOT_DRAWABLE` if the classifier isn't a tree.

```bash
curl http://localhost:7070/models/iris-j48/tree
```

```json
{
  "name": "iris-j48",
  "type": "tree",
  "format": "dot",
  "graph": "digraph J48Tree {\nN0 [label=\"petalwidth\"]\n..."
}
```

#### `GET /models/{name}/graph`

Same shape, for Bayes-net classifiers (`weka.classifiers.bayes.BayesNet`).

---

### Post-training diagnostics

All five POST endpoints share the same request shape:

| Field        | Required | Notes |
| ------------ | -------- | ----- |
| `model`      | yes      | Trained model name. |
| `dataset`    | yes      | Evaluation dataset. |
| `classValue` | no       | Nominal class label (defaults to index 0). |
| `bins`       | no       | Number of bins for calibration (default 10, max 100). |
| `sample`     | no       | Max points to return for `/errors` (default 500, max 5000). |
| `seed`       | no       | Random seed for sampling (default 42). |

#### `POST /diagnostics/errors`

Per-instance predicted vs. actual — Weka's "Visualize classifier errors".

```bash
curl -X POST http://localhost:7070/diagnostics/errors \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-j48","dataset":"iris"}'
```

```json
{
  "model": "iris-j48", "dataset": "iris",
  "classType": "nominal",
  "totalInstances": 150, "sampled": 150,
  "points": [
    { "index": 0, "actual": "Iris-setosa", "predicted": "Iris-setosa", "correct": true }
  ]
}
```

For numeric class: `points[i]` has `actual`, `predicted`, and `error` (`|p - a|`).

#### `POST /diagnostics/threshold-curve`

ROC / threshold curve for a chosen positive class. Includes AUC. Numeric-class models → 400 `NOT_NOMINAL_CLASS`.

```bash
curl -X POST http://localhost:7070/diagnostics/threshold-curve \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-j48","dataset":"iris","classValue":"Iris-versicolor"}'
```

```json
{
  "model": "iris-j48", "dataset": "iris",
  "classValue": "Iris-versicolor",
  "auc": 0.99,
  "points": [
    { "threshold": 0.0, "truePositiveRate": 1.0, "falsePositiveRate": 1.0,
      "precision": 0.33, "recall": 1.0, "fMeasure": 0.5 }
  ]
}
```

#### `POST /diagnostics/margin-curve`

Cumulative margin distribution — Weka's `MarginCurve`. Useful for ensemble diagnostics (AdaBoost, Bagging) to see whether margins improve as boosting iterations accrue. Numeric-class models → 400 `NOT_NOMINAL_CLASS`.

```bash
curl -X POST http://localhost:7070/diagnostics/margin-curve \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-j48","dataset":"iris"}'
```

```json
{
  "model": "iris-j48",
  "dataset": "iris",
  "points": [
    { "margin": -1.0, "current": 0.0, "cumulative": 0.0 },
    { "margin": 0.92, "current": 3.0, "cumulative": 0.02 },
    { "margin": 1.0,  "current": 147.0, "cumulative": 1.0 }
  ]
}
```

Field meaning: `margin` is the difference between the probability assigned to the actual class and the highest probability for any other class. `current` and `cumulative` together describe the empirical CDF of margins.

#### `POST /diagnostics/cost-curve`

Drummond–Holte cost curve for a chosen positive class. Plots the *Normalized Expected Cost* against the *Probability Cost Function* — i.e. how sensitive the model's expected cost is to changes in class skew. Numeric-class models → 400 `NOT_NOMINAL_CLASS`.

```bash
curl -X POST http://localhost:7070/diagnostics/cost-curve \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-j48","dataset":"iris","classValue":"Iris-versicolor"}'
```

```json
{
  "model": "iris-j48",
  "dataset": "iris",
  "classValue": "Iris-versicolor",
  "points": [
    { "probabilityCostFunction": 0.0, "normalizedExpectedCost": 0.0 },
    { "probabilityCostFunction": 0.5, "normalizedExpectedCost": 0.04 },
    { "probabilityCostFunction": 1.0, "normalizedExpectedCost": 0.0 }
  ]
}
```

Plot `probabilityCostFunction` on the x-axis and `normalizedExpectedCost` on the y-axis. A flat curve near zero means the model is robust to class skew; a curve that spikes near the centre means cost is highly sensitive.

#### `POST /diagnostics/calibration`

Reliability diagram + Brier score. Manually binned because Weka doesn't ship a first-party utility.

```bash
curl -X POST http://localhost:7070/diagnostics/calibration \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-j48","dataset":"iris","classValue":"Iris-versicolor","bins":10}'
```

```json
{
  "model": "iris-j48", "dataset": "iris",
  "classValue": "Iris-versicolor",
  "brierScore": 0.04,
  "totalInstances": 150,
  "bins": [
    { "bin": 0, "lo": 0.0, "hi": 0.1, "count": 99, "predictedProb": 0.01, "observedFraction": 0.0 }
  ]
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
| `INVALID_FILTER`      | 400  | Filter outside `weka.filters.` allowlist or unknown class |
| `TRANSFORM_FAILED`    | 422  | Weka threw applying a filter chain                   |
| `INVALID_ATTRIBUTE`   | 400  | Attribute name not on dataset                        |
| `INVALID_CLASS_VALUE` | 400  | Class value not in the class attribute's domain      |
| `NOT_DRAWABLE`        | 400  | Classifier doesn't implement Drawable / wrong graph type |
| `NOT_NOMINAL_CLASS`   | 400  | Diagnostic requires a nominal class but the model's class is numeric |
| `NOT_NUMERIC_CLASS`   | 400  | Reserved for future numeric-class diagnostics        |
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

# 4a. EDA on the dataset
curl 'http://localhost:7070/datasets/iris/summary'
curl 'http://localhost:7070/datasets/iris/histogram?attribute=petallength&bins=10&groupBy=class'
curl 'http://localhost:7070/datasets/iris/scatter?x=petallength&y=petalwidth'

# 4b. apply a filter to produce a new dataset
curl -X POST http://localhost:7070/transform \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"iris","filters":[{"filter":"weka.filters.unsupervised.attribute.Normalize","options":[]}],"outputName":"iris-norm"}'

# 4c. post-training diagnostics
curl -X POST http://localhost:7070/diagnostics/threshold-curve \
  -H 'Content-Type: application/json' \
  -d '{"model":"iris-j48","dataset":"iris","classValue":"Iris-versicolor"}'
curl http://localhost:7070/models/iris-j48/tree

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
