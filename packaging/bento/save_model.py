"""Enregistre le modèle MLflow dans le catalogue local BentoML."""

import json
import os
import sys
from pathlib import Path

import bentoml
import bentoml.sklearn
import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataset_config import load_config


def main():
    cfg = load_config(ROOT / "dataset.yaml")

    metadata = json.loads(
        (ROOT / "packaging/model/metadata.json").read_text(
            encoding="utf-8"
        )
    )

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    model_uri = (
        f"models:/{metadata['model_name']}/"
        f"{metadata['model_version']}"
    )

    print(f"Chargement : {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    saved = bentoml.sklearn.save_model(
        "scoring_model",
        model,
        signatures={
            "predict": {"batchable": False},
            "predict_proba": {"batchable": False},
        },
        labels={
            "mlflow_model": metadata["model_name"],
            "mlflow_version": str(metadata["model_version"]),
        },
        metadata={
            "dataset": cfg.name,
            "numeric": cfg.numeric,
            "categorical": cfg.categorical,
            "positive_class": str(cfg.positive_class),
            "threshold": metadata["threshold"],
            "model_version": str(metadata["model_version"]),
            "run_id": metadata["run_id"],
        },
    )

    print(f"Modèle enregistré dans BentoML : {saved.tag}")


if __name__ == "__main__":
    main()
