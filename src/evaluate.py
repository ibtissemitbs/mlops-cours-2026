"""Évalue le modèle local et écrit reports/metrics.json."""

import json
from pathlib import Path

import mlflow.sklearn
from mlflow.models import Model
from sklearn.metrics import confusion_matrix

from src.dataset_config import load_config
from src.train import MODEL_DIR, compute_metrics

TEST_CSV = Path("data/processed/test.csv")
METRICS_JSON = Path("reports/metrics.json")


def main():
    cfg = load_config()
    X, y = cfg.read_xy(TEST_CSV)

    model = mlflow.sklearn.load_model(str(MODEL_DIR))
    metadata = Model.load(str(MODEL_DIR)).metadata
    threshold = float(metadata["decision_threshold"])

    proba = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(cfg, y, proba, threshold)

    tn, fp, fn, tp = confusion_matrix(y, (proba >= threshold).astype(int), labels=[0, 1]).ravel()

    report = {
        **{key: round(float(value), 4) for key, value in metrics.items()},
        "decision_threshold": threshold,
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "n_test": int(len(y)),
    }

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ROC AUC test : {report['roc_auc']} -> {METRICS_JSON}")


if __name__ == "__main__":
    main()
