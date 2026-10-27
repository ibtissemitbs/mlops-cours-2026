"""Composants Kubeflow du projet de maintenance prédictive."""

import os

from kfp import dsl
from kfp.dsl import (
    Artifact,
    ClassificationMetrics,
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
)

PIPELINE_IMAGE = os.getenv(
    "PIPELINE_IMAGE",
    "localhost:5001/scoring-pipeline:1.0.0",
)
MLFLOW_URI = "http://mlflow.mlflow:5000"


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def ingest_data(raw_data: Output[Dataset]) -> None:
    """Télécharge les données avec le script du Lab 1."""
    import pandas as pd

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.train import file_md5

    cfg = load_config()
    run_stage(
        "src.download",
        outputs={cfg.raw_path: raw_data.path},
    )
    raw_data.metadata["dataset"] = cfg.name
    raw_data.metadata["rows"] = len(pd.read_csv(raw_data.path))
    raw_data.metadata["md5"] = file_md5(raw_data.path)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def validate_data(
    raw_data: Input[Dataset],
    report: Output[Artifact],
    checks: Output[Metrics],
) -> None:
    """Arrête le pipeline si les données ne respectent pas le contrat."""
    import json
    from pathlib import Path

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.validate_data import REPORT_JSON

    cfg = load_config()
    run_stage(
        "src.validate_data",
        inputs={cfg.raw_path: raw_data.path},
        outputs={str(REPORT_JSON): report.path},
    )
    summary = json.loads(Path(report.path).read_text())
    for name in ("n_checks", "n_failed", "rows", "positive_rate"):
        checks.log_metric(name, summary[name])


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def prepare_features(
    raw_data: Input[Dataset],
    train_data: Output[Dataset],
    validation_data: Output[Dataset],
    test_data: Output[Dataset],
) -> None:
    """Prépare les trois ensembles avec src.prepare."""
    import pandas as pd

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.train import TEST_CSV, TRAIN_CSV, VALIDATION_CSV

    cfg = load_config()
    run_stage(
        "src.prepare",
        inputs={cfg.raw_path: raw_data.path},
        outputs={
            str(TRAIN_CSV): train_data.path,
            str(VALIDATION_CSV): validation_data.path,
            str(TEST_CSV): test_data.path,
        },
    )
    for split in (train_data, validation_data, test_data):
        y = cfg.encode_target(pd.read_csv(split.path)[cfg.target])
        split.metadata["rows"] = len(y)
        split.metadata["positive_rate"] = round(float(y.mean()), 4)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def train_model(
    raw_data: Input[Dataset],
    train_data: Input[Dataset],
    validation_data: Input[Dataset],
    model: Output[Model],
    model_type: str,
    C: float,
    class_weight: str,
    n_estimators: int,
    max_depth: str,
    learning_rate: float,
    random_state: int,
    mlflow_uri: str,
    kfp_run_id: str,
    hidden_layer_sizes: str = "32",
    alpha: float = 0.0001,
    learning_rate_init: float = 0.001,
    max_iter: int = 500,
) -> str:
    """Entraîne sur train et mesure les scores de validation."""
    from mlflow import MlflowClient
    from mlflow.models import Model as MlflowModel

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.train import MODEL_DIR, TRAIN_CSV, VALIDATION_CSV

    cfg = load_config()
    run_stage(
        "src.train",
        args=[
            "--model",
            model_type,
            "--C",
            C,
            "--class_weight",
            class_weight,
            "--n_estimators",
            n_estimators,
            "--max_depth",
            max_depth,
            "--learning_rate",
            learning_rate,
            "--random_state",
            random_state,
            "--hidden_layer_sizes",
            hidden_layer_sizes,
            "--alpha",
            alpha,
            "--learning_rate_init",
            learning_rate_init,
            "--max_iter",
            max_iter,
        ],
        inputs={
            cfg.raw_path: raw_data.path,
            str(TRAIN_CSV): train_data.path,
            str(VALIDATION_CSV): validation_data.path,
        },
        outputs={str(MODEL_DIR): model.path},
        env={
            "MLFLOW_TRACKING_URI": mlflow_uri,
            "LAUNCHED_BY": "kfp",
        },
    )
    metadata = MlflowModel.load(model.path).metadata
    run_id = metadata["mlflow_run_id"]
    MlflowClient(mlflow_uri).set_tag(run_id, "kfp_run_id", kfp_run_id)
    model.metadata["mlflow_run_id"] = run_id
    model.metadata["model_type"] = model_type
    model.metadata["decision_threshold"] = round(float(metadata["decision_threshold"]), 4)
    return run_id


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    report: Output[Artifact],
    metrics: Output[Metrics],
    confusion: Output[ClassificationMetrics],
    run_id: str,
    mlflow_uri: str,
) -> float:
    """Évalue sur test sans écraser les scores de validation MLflow."""
    import json
    from pathlib import Path

    import pandas as pd
    from mlflow import MlflowClient

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.evaluate import METRICS_JSON, TEST_CSV
    from src.train import MODEL_DIR

    cfg = load_config()
    run_stage(
        "src.evaluate",
        inputs={
            str(MODEL_DIR): model.path,
            str(TEST_CSV): test_data.path,
        },
        outputs={str(METRICS_JSON): report.path},
        env={"MLFLOW_TRACKING_URI": mlflow_uri},
    )
    results = json.loads(Path(report.path).read_text())

    client = MlflowClient(mlflow_uri)
    for name, value in results.items():
        if isinstance(value, (int, float)):
            metrics.log_metric(name, value)
            client.log_metric(run_id, f"test_{name}", float(value))

    positive = str(cfg.positive_class)
    others = sorted(set(pd.read_csv(test_data.path)[cfg.target].astype(str)) - {positive})
    negative = others[0] if len(others) == 1 else f"non {positive}"
    c = results["confusion"]
    confusion.log_confusion_matrix(
        [negative, positive],
        [[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
    )
    client.log_dict(run_id, results, str(METRICS_JSON))
    return float(results[cfg.primary_metric])


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def register_model(run_id: str, mlflow_uri: str) -> str:
    """Enregistre le modèle avec l'alias challenger."""
    from mlflow import MlflowClient

    from pipelines.stage import run_stage
    from src.promote import MODEL_NAME

    run_stage(
        "src.promote",
        args=["--run-id", run_id],
        env={"MLFLOW_TRACKING_URI": mlflow_uri},
    )
    versions = MlflowClient(mlflow_uri).search_model_versions(
        f"name = '{MODEL_NAME}' and run_id = '{run_id}'"
    )
    if not versions:
        raise RuntimeError("Modèle absent du Registry après enregistrement.")
    return str(versions[0].version)
