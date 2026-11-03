"""Récupère le champion MLflow et l'exporte au format ONNX."""

import copy
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from skl2onnx import to_onnx
from skl2onnx.common.data_types import DoubleTensorType, StringTensorType
from skl2onnx.sklapi import CastTransformer
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataset_config import load_config

MODEL_NAME = os.getenv("MODEL_NAME", "scoring-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_DIR = ROOT / "packaging" / "model"
TARGET_OPSET = 17


def onnx_ready(model, cfg):
    """Adapte une copie du prétraitement pour la conversion ONNX."""
    pipeline = copy.deepcopy(model)

    pre = next(
        step
        for _, step in pipeline.steps
        if isinstance(step, ColumnTransformer)
    )

    if cfg.categorical:
        categorical = pre.named_transformers_["cat"]
        categorical.named_steps["impute"].missing_values = ""

    if cfg.numeric:
        numeric = pre.named_transformers_["num"]
        imputer = numeric.named_steps["impute"]
        scaler = numeric.named_steps["scale"]

        # Conserver la précision du calcul avant l'entrée dans les arbres.
        imputer.statistics_ = (
            imputer.statistics_ - scaler.mean_
        ) / scaler.scale_

        cast = CastTransformer(dtype=np.float32).fit(
            np.zeros((1, len(cfg.numeric)))
        )

        numeric.steps = [
            ("scale", scaler),
            ("float32", cast),
            ("impute", imputer),
        ]

    return pipeline


def main():
    cfg = load_config(ROOT / "dataset.yaml")

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    client = MlflowClient()

    version = client.get_model_version_by_alias(
        MODEL_NAME, MODEL_ALIAS
    )
    model_uri = f"models:/{MODEL_NAME}/{version.version}"
    model = mlflow.sklearn.load_model(model_uri)

    print(f"Modèle chargé : {model_uri}")
    print(f"Algorithme : {type(model.steps[-1][1]).__name__}")

    initial_types = [
        (column, DoubleTensorType([None, 1]))
        for column in cfg.numeric
    ] + [
        (column, StringTensorType([None, 1]))
        for column in cfg.categorical
    ]

    onnx_model = to_onnx(
        onnx_ready(model, cfg),
        initial_types=initial_types,
        target_opset=TARGET_OPSET,
        options={"zipmap": False},
    )

    run = client.get_run(version.run_id)
    logged_threshold = run.data.params.get("decision_threshold")

    if logged_threshold is not None:
        threshold = float(logged_threshold)
        threshold_source = "param decision_threshold du run"
    else:
        weight_ratio = 1.0
        estimator = model.steps[-1][1]

        if getattr(estimator, "class_weight", None) == "balanced":
            _, y = cfg.read_xy(ROOT / "data/processed/train.csv")
            weight_ratio = float((y == 0).sum() / (y == 1).sum())

        threshold = cfg.cost_threshold(weight_ratio)
        threshold_source = "cfg.cost_threshold (repli)"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    onnx_bytes = onnx_model.SerializeToString()
    onnx_path = MODEL_DIR / "model.onnx"
    onnx_path.write_bytes(onnx_bytes)

    metadata = {
        "dataset": cfg.name,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": str(version.version),
        "run_id": version.run_id,
        "target": cfg.target,
        "positive_class": str(cfg.positive_class),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "features": {
            "numeric": cfg.numeric,
            "categorical": cfg.categorical,
        },
        "onnx_opset": TARGET_OPSET,
        "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    metadata_path = MODEL_DIR / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"ONNX exporté : {onnx_path.relative_to(ROOT)}")
    print(f"Taille : {len(onnx_bytes) / 1024:.1f} Ko")
    print(f"Seuil de décision : {threshold}")
    print(f"Métadonnées : {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
