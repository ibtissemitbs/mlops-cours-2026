"""Compare les prédictions ONNX avec celles du modèle scikit-learn."""

import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataset_config import load_config


def main():
    cfg = load_config(ROOT / "dataset.yaml")
    model_dir = ROOT / "packaging/model"

    metadata = json.loads(
        (model_dir / "metadata.json").read_text(encoding="utf-8")
    )
    threshold = float(metadata["threshold"])

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    # Charger exactement la version utilisée pendant l'export.
    model_uri = (
        f"models:/{metadata['model_name']}/"
        f"{metadata['model_version']}"
    )
    model = mlflow.sklearn.load_model(model_uri)

    X, _ = cfg.read_xy(ROOT / "data/processed/test.csv")
    sklearn_proba = model.predict_proba(X)[:, 1]

    session = ort.InferenceSession(
        str(model_dir / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    # L'export utilise cet ordre : numériques, puis catégorielles.
    # Les noms réels ONNX peuvent contenir des "_" à la place des espaces.
    columns = [*cfg.numeric, *cfg.categorical]
    onnx_inputs = session.get_inputs()

    if len(onnx_inputs) != len(columns):
        raise ValueError(
            "Le nombre d'entrées ONNX ne correspond pas au contrat."
        )

    inputs = {}

    for column, onnx_input in zip(columns, onnx_inputs):
        if column in cfg.numeric:
            values = X[column].to_numpy(dtype=np.float64)
        else:
            values = (
                X[column]
                .fillna("")
                .astype(str)
                .to_numpy()
            )

        inputs[onnx_input.name] = values.reshape(-1, 1)
        print(f"{column} → {onnx_input.name}")

    outputs = session.run(None, inputs)
    onnx_proba = np.asarray(outputs[1])[:, 1]

    differences = np.abs(sklearn_proba - onnx_proba)

    sklearn_predictions = sklearn_proba >= threshold
    onnx_predictions = onnx_proba >= threshold

    disagreements = int(
        np.count_nonzero(
            sklearn_predictions != onnx_predictions
        )
    )

    print(f"\nLignes comparées : {len(X)}")
    print(
        f"Écart maximal des probabilités : "
        f"{differences.max():.8f}"
    )
    print(
        f"Écart moyen des probabilités : "
        f"{differences.mean():.8f}"
    )
    print(f"Décisions différentes : {disagreements}")
    print(f"Seuil de décision : {threshold}")

    np.testing.assert_allclose(
        onnx_proba,
        sklearn_proba,
        atol=1e-4,
        rtol=0,
        err_msg=(
            "Les probabilités ONNX diffèrent trop "
            "de scikit-learn."
        ),
    )

    assert disagreements == 0, (
        f"{disagreements} décisions différentes après conversion."
    )

    print("Vérification réussie : conversion ONNX fidèle.")


if __name__ == "__main__":
    main()
