"""Crée deux fichiers pickle pour la démonstration ModelScan.

Ne pas charger ces fichiers : on les analyse uniquement avec ModelScan.
"""

import json
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Payload:
    """Instruction de démonstration détectable par ModelScan."""

    def __reduce__(self):
        return (
            os.system,
            ("echo 'Demonstration pedagogique ModelScan'",),
        )


def main():
    metadata = json.loads(
        (ROOT / "packaging/model/metadata.json").read_text(
            encoding="utf-8"
        )
    )

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    model = mlflow.sklearn.load_model(
        f"models:/{metadata['model_name']}/"
        f"{metadata['model_version']}"
    )

    with (HERE / "model.pkl").open("wb") as file:
        pickle.dump(model, file)

    with (HERE / "malicious.pkl").open("wb") as file:
        pickle.dump(
            {"model": model, "metadata": Payload()},
            file,
        )

    print("Créés : model.pkl et malicious.pkl")
    print("Ne pas les charger ; utiliser ModelScan uniquement.")


if __name__ == "__main__":
    main()
