"""Valide le modèle avant son enregistrement : Evidently et pytest."""

import json
from pathlib import Path
from xml.etree import ElementTree

import mlflow.sklearn
import pandas as pd
import pytest
from mlflow.models import Model

from src.dataset_config import load_config
from src.train import MODEL_DIR, TEST_CSV, TRAIN_CSV, compute_metrics

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_TESTS = ROOT / "tests" / "test_model_behavior.py"

REPORT_JSON = Path("reports/model_validation.json")
REPORT_MD = Path("reports/model_validation.md")
REPORT_HTML = Path("reports/model_validation.html")
JUNIT_XML = Path("reports/model_behavior.xml")

MAX_TRAIN_TEST_GAP = 0.2

EVIDENTLY_NAMES = {
    "roc_auc": "rocauc",
    "f1": "f1score",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}

PRESET_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1score",
    "rocauc",
    "logloss",
    "tpr",
    "tnr",
    "fpr",
    "fnr",
]


def positive_probability(model, X):
    """Le modèle est entraîné avec une cible encodée en 0/1."""
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, positive_index]


def evidently_checks(cfg, model, threshold: float) -> list:
    """Génère le rapport HTML et récupère les résultats des tests."""
    from evidently import BinaryClassification, DataDefinition, Dataset, Report
    from evidently.presets import ClassificationPreset
    from evidently.tests import Reference, eq, gte

    definition = DataDefinition(
        classification=[
            BinaryClassification(
                target="target",
                prediction_probas="proba",
                pos_label=1,
            )
        ]
    )

    def dataset(path):
        X, y = cfg.read_xy(path)
        frame = pd.DataFrame({
            "target": y.to_numpy(),
            "proba": positive_probability(model, X),
        })
        return Dataset.from_pandas(
            frame,
            data_definition=definition,
        )

    # Désactiver les tests automatiques pour utiliser ceux du lab.
    tests = {
        f"{name}_tests": []
        for name in PRESET_METRICS
    }

    # Comparer la ROC AUC de test à celle d'entraînement.
    tests["rocauc_tests"] = [
        eq(Reference(relative=MAX_TRAIN_TEST_GAP))
    ]

    # Exiger la performance minimale de dataset.yaml.
    if cfg.primary_metric not in EVIDENTLY_NAMES:
        raise ValueError(
            f"Métrique non prise en charge : {cfg.primary_metric}"
        )

    metric_name = EVIDENTLY_NAMES[cfg.primary_metric]
    tests[f"{metric_name}_tests"].append(
        gte(cfg.promotion_threshold)
    )

    report = Report([
        ClassificationPreset(
            probas_threshold=threshold,
            **tests,
        )
    ])

    snapshot = report.run(
        dataset(TEST_CSV),
        dataset(TRAIN_CSV),
    )

    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_HTML))

    results = []
    for test in snapshot.dict()["tests"]:
        status = getattr(test["status"], "value", test["status"])
        results.append({
            "name": test["name"],
            "success": status == "SUCCESS",
        })

    if not results:
        raise RuntimeError("Aucun test Evidently n'a été exécuté.")

    return results


def behavior_checks() -> list:
    """Exécute les tests comportementaux et lit le rapport JUnit."""
    JUNIT_XML.parent.mkdir(parents=True, exist_ok=True)

    code = pytest.main([
        "-q",
        "-p",
        "no:cacheprovider",
        "-W",
        "ignore::pytest.PytestAssertRewriteWarning",
        str(BEHAVIOR_TESTS),
        f"--junitxml={JUNIT_XML}",
    ])

    if code not in (
        pytest.ExitCode.OK,
        pytest.ExitCode.TESTS_FAILED,
    ):
        return [{
            "name": f"pytest : code retour {int(code)}",
            "success": False,
            "skipped": False,
        }]

    results = []
    root = ElementTree.parse(JUNIT_XML).getroot()

    for case in root.iter("testcase"):
        skipped = case.find("skipped") is not None
        results.append({
            "name": case.get("name"),
            "success": (
                case.find("failure") is None
                and case.find("error") is None
            ),
            "skipped": skipped,
        })

    if not any(not item["skipped"] for item in results):
        return [{
            "name": "Aucun test comportemental exécuté",
            "success": False,
            "skipped": False,
        }]

    return results


def markdown(summary: dict) -> str:
    lines = [
        f"### Validation du modèle : {summary['n_failed']} échec(s) "
        f"sur {summary['n_checks']} contrôles listés",
        "",
        "| Famille | Test | Statut |",
        "|---|---|---|",
    ]

    for family in ("evidently", "behavior"):
        for check in summary[family]:
            if check.get("skipped"):
                status = "IGNORÉ"
            else:
                status = "OK" if check["success"] else "ÉCHEC"

            name = str(check["name"]).replace("|", "/")
            lines.append(f"| {family} | {name} | {status} |")

    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = load_config()
    model = mlflow.sklearn.load_model(str(MODEL_DIR))
    metadata = Model.load(str(MODEL_DIR)).metadata
    threshold = float(metadata["decision_threshold"])

    X, y = cfg.read_xy(TEST_CSV)
    metrics = compute_metrics(
        cfg,
        y,
        positive_probability(model, X),
        threshold,
    )

    evidently = evidently_checks(cfg, model, threshold)
    behavior = behavior_checks()
    checks = [*evidently, *behavior]
    failed = [
        check["name"] for check in checks
        if not check["success"]
    ]

    summary = {
        "success": not failed,
        "n_checks": len(checks),
        "n_skipped": sum(
            bool(check.get("skipped")) for check in checks
        ),
        "n_failed": len(failed),
        "failed": failed,
        cfg.primary_metric: round(
            float(metrics[cfg.primary_metric]), 4
        ),
        "decision_threshold": threshold,
        "evidently": evidently,
        "behavior": behavior,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(
        markdown(summary),
        encoding="utf-8",
    )

    print(markdown(summary))
    print("Modèle VALIDÉ" if not failed else f"Modèle REFUSÉ : {failed}")
    print(f"Rapport HTML : {REPORT_HTML}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
