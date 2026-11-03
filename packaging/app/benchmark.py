"""Compare la latence de scikit-learn et ONNX Runtime."""

import json
import os
import platform
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataset_config import load_config

N = 1000


def bench(name, function):
    # Chauffe : les premiers appels ne sont pas mesurés.
    for _ in range(10):
        function()

    durations = []

    for _ in range(N):
        start = time.perf_counter()
        function()
        durations.append(
            (time.perf_counter() - start) * 1000
        )

    return {
        "name": name,
        "p50": float(np.percentile(durations, 50)),
        "p95": float(np.percentile(durations, 95)),
        "p99": float(np.percentile(durations, 99)),
    }


def main():
    cfg = load_config(ROOT / "dataset.yaml")

    metadata = json.loads(
        (ROOT / "packaging/model/metadata.json").read_text(
            encoding="utf-8"
        )
    )

    X, _ = cfg.read_xy(ROOT / "data/processed/test.csv")

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    model = mlflow.sklearn.load_model(
        f"models:/{metadata['model_name']}/"
        f"{metadata['model_version']}"
    )

    session = ort.InferenceSession(
        str(ROOT / "packaging/model/model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    columns = [*cfg.numeric, *cfg.categorical]
    onnx_inputs = session.get_inputs()

    if len(columns) != len(onnx_inputs):
        raise ValueError(
            "Les entrées ONNX ne correspondent pas au contrat."
        )

    input_names = {
        column: entry.name
        for column, entry in zip(columns, onnx_inputs)
    }

    def onnx_feed(frame):
        inputs = {}

        for column in cfg.numeric:
            inputs[input_names[column]] = (
                frame[column]
                .to_numpy(dtype=np.float64)
                .reshape(-1, 1)
            )

        for column in cfg.categorical:
            inputs[input_names[column]] = (
                frame[column]
                .fillna("")
                .astype(str)
                .to_numpy(dtype=object)
                .reshape(-1, 1)
            )

        return inputs

    one_row = X.head(1)
    batch = X.head(100)

    # Préparer les entrées hors du chronométrage.
    one_feed = onnx_feed(one_row)
    batch_feed = onnx_feed(batch)

    results = [
        bench(
            "sklearn_pipeline",
            lambda: model.predict_proba(one_row),
        ),
        bench(
            "onnx_python",
            lambda: session.run(None, one_feed),
        ),
        bench(
            f"onnx_batch_{len(batch)}",
            lambda: session.run(None, batch_feed),
        ),
    ]

    table = [
        "| Backend | P50 (ms) | P95 (ms) | P99 (ms) |",
        "|---|---|---|---|",
    ]

    for result in results:
        table.append(
            f"| {result['name']} "
            f"| {result['p50']:.4f} "
            f"| {result['p95']:.4f} "
            f"| {result['p99']:.4f} |"
        )

    speedup = results[0]["p50"] / results[1]["p50"]

    print("\n".join(table))
    print(
        f"\nAccélération ONNX sur une ligne, en P50 : "
        f"x{speedup:.1f}"
    )

    report = ROOT / "reports/benchmark.md"
    report.parent.mkdir(parents=True, exist_ok=True)

    report.write_text(
        f"# Benchmark — {cfg.name}\n\n"
        f"Modèle : version {metadata['model_version']}\n\n"
        f"{N} appels par scénario, après 10 appels de chauffe.\n\n"
        f"Python : {platform.python_version()} ; "
        f"architecture : {platform.machine()}.\n\n"
        + "\n".join(table)
        + f"\n\nAccélération ONNX en P50 sur une ligne : "
        f"x{speedup:.1f}.\n\n"
        "Le scénario batch mesure le temps total pour 100 lignes.\n"
        "Les mesures excluent le réseau HTTP et le chargement "
        "des modèles. Les entrées ONNX sont préparées à l'avance ; "
        "le pipeline scikit-learn reçoit un DataFrame.\n",
        encoding="utf-8",
    )

    print(f"Rapport : {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
