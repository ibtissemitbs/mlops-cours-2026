"""Enregistre un modèle existant dans le Registry MLflow."""

import argparse
import os

import mlflow

from src.dataset_config import load_config
from src.train import EXPERIMENT

MODEL_NAME = "scoring-model"
PROTOCOL = "train6400-val1600-test2000"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Run précis à enregistrer")
    parser.add_argument("--filter", default="", help="Filtre MLflow supplémentaire")
    parser.add_argument(
        "--champion",
        action="store_true",
        help="Attribuer champion plutôt que challenger",
    )
    args = parser.parse_args()

    cfg = load_config()
    metric = cfg.primary_metric
    if metric not in {
        "roc_auc",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "accuracy",
    }:
        raise SystemExit("Métrique principale non prise en charge.")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise SystemExit("Expérience introuvable.")

    filters = (
        "attributes.status = 'FINISHED' "
        "AND tags.evaluation_split = 'validation' "
        f"AND tags.protocol = '{PROTOCOL}' "
        f"AND tags.dataset = '{cfg.name}'"
    )

    # Le meilleur candidat est déterminé sur la validation uniquement.
    best = client.search_runs(
        [exp.experiment_id],
        filter_string=filters,
        order_by=[f"metrics.{metric} DESC", "attributes.start_time DESC"],
        max_results=1,
    )
    if not best:
        raise SystemExit("Aucun candidat de validation trouvé.")

    if args.run_id:
        run = client.get_run(args.run_id)
    else:
        candidates = client.search_runs(
            [exp.experiment_id],
            filter_string=(filters + f" AND {args.filter}" if args.filter else filters),
            order_by=[f"metrics.{metric} DESC", "attributes.start_time DESC"],
            max_results=1,
        )
        if not candidates:
            raise SystemExit("Aucun run ne correspond au filtre.")
        run = client.get_run(candidates[0].info.run_id)

    tags = run.data.tags
    if (
        run.info.status != "FINISHED"
        or run.info.experiment_id != exp.experiment_id
        or tags.get("dataset") != cfg.name
        or tags.get("evaluation_split") != "validation"
        or tags.get("protocol") != PROTOCOL
    ):
        raise SystemExit("Le run ne correspond pas au protocole attendu.")

    score = run.data.metrics.get(metric)
    if score is None:
        raise SystemExit("Métrique de validation absente.")

    if args.champion:
        if run.info.run_id != best[0].info.run_id:
            raise SystemExit("Champion doit être le meilleur run de validation.")
        if score < cfg.promotion_threshold:
            raise SystemExit("Seuil de qualité non atteint.")

    existing = client.search_model_versions(
        f"name = '{MODEL_NAME}' and run_id = '{run.info.run_id}'"
    )
    if existing:
        version = existing[0].version
        print(f"Version déjà existante : {version}")
    else:
        outputs = run.outputs.model_outputs if run.outputs else []
        if len(outputs) != 1:
            raise SystemExit("Impossible d'identifier un modèle unique.")

        model_id = outputs[0].model_id
        registered = mlflow.register_model(f"models:/{model_id}", MODEL_NAME)
        version = registered.version

    version_tags = {
        "dataset": cfg.name,
        "model_type": tags.get("model_type", "unknown"),
        "selection_split": "validation",
        f"validation_{metric}": str(score),
        "validation_status": ("passed" if score >= cfg.promotion_threshold else "failed"),
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(MODEL_NAME, version, key, value)

    alias = "champion" if args.champion else "challenger"
    client.set_registered_model_alias(MODEL_NAME, alias, version)

    print(f"Modèle : {run.info.run_name}")
    print(f"Validation {metric} : {score:.4f}")
    print(f"Registry : {MODEL_NAME}, version {version}")
    print(f"Alias : @{alias}")


if __name__ == "__main__":
    main()
