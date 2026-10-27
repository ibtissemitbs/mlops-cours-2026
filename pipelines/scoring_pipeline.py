"""Pipeline Kubeflow adapté au dataset AI4I."""

from pathlib import Path

import yaml
from kfp import compiler, dsl, kubernetes

from pipelines.components import (
    MLFLOW_URI,
    evaluate_model,
    ingest_data,
    prepare_features,
    register_model,
    train_model,
    validate_data,
)

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))["train"]


@dsl.pipeline(
    name="scoring-pipeline",
    description="Maintenance prédictive AI4I : préparation, entraînement et évaluation.",
)
def scoring_pipeline(
    model_type: str = PARAMS["model"],
    C: float = float(PARAMS["C"]),
    class_weight: str = str(PARAMS["class_weight"]),
    n_estimators: int = int(PARAMS["n_estimators"]),
    max_depth: str = str(PARAMS["max_depth"]),
    learning_rate: float = float(PARAMS["learning_rate"]),
    random_state: int = int(PARAMS["random_state"]),
    hidden_layer_sizes: str = ",".join(str(n) for n in PARAMS["hidden_layer_sizes"]),
    alpha: float = float(PARAMS["alpha"]),
    learning_rate_init: float = float(PARAMS["learning_rate_init"]),
    max_iter: int = int(PARAMS["max_iter"]),
    mlflow_uri: str = MLFLOW_URI,
):
    ingest = ingest_data()
    ingest.set_display_name("1 - Télécharger AI4I")

    validation = validate_data(
        raw_data=ingest.outputs["raw_data"],
    )
    validation.set_display_name("2 - Valider les données")

    prepare = prepare_features(
        raw_data=ingest.outputs["raw_data"],
    )
    prepare.after(validation)
    prepare.set_display_name("3 - Préparer train, validation et test")

    train = train_model(
        raw_data=ingest.outputs["raw_data"],
        train_data=prepare.outputs["train_data"],
        validation_data=prepare.outputs["validation_data"],
        model_type=model_type,
        C=C,
        class_weight=class_weight,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        mlflow_uri=mlflow_uri,
        kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    )
    train.set_display_name("4 - Entraîner et mesurer sur validation")

    evaluate = evaluate_model(
        model=train.outputs["model"],
        test_data=prepare.outputs["test_data"],
        run_id=train.outputs["Output"],
        mlflow_uri=mlflow_uri,
    )
    evaluate.set_display_name("5 - Évaluer sur test")

    register = register_model(
        run_id=train.outputs["Output"],
        mlflow_uri=mlflow_uri,
    )
    register.after(evaluate)
    register.set_display_name("6 - Enregistrer le modèle")
    register.set_caching_options(False)
    # Téléchargement : réessayer en cas d'échec réseau.
    ingest.set_retry(num_retries=3, backoff_duration="30s")
    ingest.set_cpu_limit("500m")
    ingest.set_memory_limit("512Mi")
    kubernetes.set_timeout(ingest, 600)

    # Validation et préparation.
    validation.set_memory_limit("512Mi")
    prepare.set_memory_limit("1Gi")

    # Réutiliser les résultats des étapes de données.
    for task in (ingest, validation, prepare):
        task.set_caching_options(True)

    # Entraînement : ressources et durée maximale.
    train.set_cpu_request("500m")
    train.set_cpu_limit("2")
    train.set_memory_request("512Mi")
    train.set_memory_limit("2Gi")
    kubernetes.set_timeout(train, 1800)

    # Produire un nouvel entraînement à chaque run.
    train.set_caching_options(False)

    # Évaluation et enregistrement.
    evaluate.set_memory_limit("1Gi")
    register.set_memory_limit("1Gi")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=scoring_pipeline,
        package_path="scoring_pipeline.yaml",
    )
    print("Pipeline compilé : scoring_pipeline.yaml")
