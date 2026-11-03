"""API FastAPI de prédiction avec le modèle ONNX."""

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, create_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataset_config import load_config

API_VERSION = "1.0.0"
MAX_BATCH = int(os.getenv("MAX_BATCH", "1000"))

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(ROOT / "packaging/model/model.onnx"),
    )
)

cfg = load_config(
    os.getenv("DATASET_CONFIG", str(ROOT / "dataset.yaml"))
)

metadata = json.loads(
    MODEL_PATH.with_name("metadata.json").read_text(encoding="utf-8")
)

THRESHOLD = float(
    os.getenv("DECISION_THRESHOLD", str(metadata["threshold"]))
)
MODEL_VERSION = str(metadata["model_version"])

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

# Même ordre que pendant l'export et la vérification ONNX.
columns = [*cfg.numeric, *cfg.categorical]
onnx_inputs = session.get_inputs()

if len(columns) != len(onnx_inputs):
    raise ValueError(
        "Le nombre d'entrées ONNX ne correspond pas au contrat."
    )

input_names = {
    column: onnx_input.name
    for column, onnx_input in zip(columns, onnx_inputs)
}

# Schéma construit à partir de dataset.yaml.
Numeric = Annotated[
    float | None,
    Field(strict=True, allow_inf_nan=False),
]
Categorical = Annotated[
    str | None,
    Field(strict=True, min_length=1),
]

Instance = create_model(
    "Instance",
    __config__=ConfigDict(extra="forbid"),
    **{
        column: (Numeric, ...)
        for column in cfg.numeric
    },
    **{
        column: (Categorical, ...)
        for column in cfg.categorical
    },
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[Instance] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH,
    )


class Prediction(BaseModel):
    probability: float
    prediction: int


class PredictResponse(BaseModel):
    model_version: str
    positive_class: str
    threshold: float
    predictions: list[Prediction]


app = FastAPI(
    title=f"Scoring API — {cfg.name}",
    version=API_VERSION,
)


def score(rows: list[dict[str, Any]]) -> np.ndarray:
    """Calcule la probabilité de panne pour chaque machine."""
    inputs = {}

    for column in cfg.numeric:
        values = [
            np.nan if row[column] is None else row[column]
            for row in rows
        ]
        inputs[input_names[column]] = np.asarray(
            values, dtype=np.float64
        ).reshape(-1, 1)

    for column in cfg.categorical:
        values = [
            "" if row[column] is None else row[column]
            for row in rows
        ]
        inputs[input_names[column]] = np.asarray(
            values, dtype=object
        ).reshape(-1, 1)

    outputs = session.run(None, inputs)
    return np.asarray(outputs[1])[:, 1]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.get("/metadata")
def get_metadata():
    model_keys = (
        "model_name",
        "model_alias",
        "model_version",
        "run_id",
        "onnx_sha256",
        "exported_at",
    )

    return {
        "api_version": API_VERSION,
        "dataset": cfg.name,
        "target": cfg.target,
        "positive_class": str(cfg.positive_class),
        "threshold": THRESHOLD,
        "features": {
            "numeric": cfg.numeric,
            "categorical": cfg.categorical,
        },
        "model": {
            key: metadata.get(key)
            for key in model_keys
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    rows = [
        instance.model_dump()
        for instance in request.instances
    ]
    probabilities = score(rows)

    return PredictResponse(
        model_version=MODEL_VERSION,
        positive_class=str(cfg.positive_class),
        threshold=THRESHOLD,
        predictions=[
            Prediction(
                probability=float(probability),
                prediction=int(probability >= THRESHOLD),
            )
            for probability in probabilities
        ],
    )
