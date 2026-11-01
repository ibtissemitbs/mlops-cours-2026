"""Pipeline Kubeflow AI4I avec validation des données et du modèle."""

from pathlib import Path

import yaml
from kfp import compiler, dsl, kubernetes

from pipelines.components import (
    MLFLOW_URI,
    detect_drift,
    evaluate_model,
    ingest_data,
    prepare_features,
    register_model,
    train_model,
    validate_data,
    validate_model,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (ROOT / "params.yaml").read_text(encoding="utf-8")
)
PARAMS = CONFIG["train"]
DRIFT = CONFIG["drift"]


@dsl.pipeline(
    name="scoring-pipeline",
    description=(
        "Maintenance prédictive AI4I : validation GX, drift, "
        "entraînement, évaluation et validation du modèle."
    ),
)
def scoring_pipeline(
    model_type: str = PARAMS["model"],
    C: float = float(PARAMS["C"]),
    class_weight: str = str(PARAMS["class_weight"]),
    n_estimators: int = int(PARAMS["n_estimators"]),
    max_depth: str = str(PARAMS["max_depth"]),
    learning_rate: float = float(PARAMS["learning_rate"]),
    random_state: int = int(PARAMS["random_state"]),
    hidden_layer_sizes: str = ",".join(
        str(n) for n in PARAMS["hidden_layer_sizes"]
    ),
    alpha: float = float(PARAMS["alpha"]),
    learning_rate_init: float = float(PARAMS["learning_rate_init"]),
    max_iter: int = int(PARAMS["max_iter"]),
    mlflow_uri: str = MLFLOW_URI,
):
    # 1. Télécharger les données.
    ingest = ingest_data()
    ingest.set_display_name("1 - Télécharger AI4I")

    # 2. Vérifier le contrat Great Expectations.
    validation = validate_data(
        raw_data=ingest.outputs["raw_data"],
    )
    validation.set_display_name("2 - Valider les données")

    # 3. Préparer les ensembles après validation des données.
    prepare = prepare_features(
        raw_data=ingest.outputs["raw_data"],
    )
    prepare.after(validation)
    prepare.set_display_name("3 - Préparer train, validation et test")

    # 4. Comparer les distributions de train et de test.
    drift_check = detect_drift(
        reference_data=prepare.outputs["train_data"],
        current_data=prepare.outputs["test_data"],
        ks_alpha=float(DRIFT["ks_alpha"]),
        psi_threshold=float(DRIFT["psi_threshold"]),
        drift_share=float(DRIFT["drift_share"]),
        fail_on_drift=True,
    )
    drift_check.set_display_name("4 - Détecter le drift")

    # 5. Entraîner seulement si le contrôle de drift réussit.
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
    train.after(drift_check)
    train.set_display_name("5 - Entraîner et mesurer sur validation")

    # 6. Évaluer le modèle sur le jeu de test.
    evaluate = evaluate_model(
        model=train.outputs["model"],
        test_data=prepare.outputs["test_data"],
        run_id=train.outputs["Output"],
        mlflow_uri=mlflow_uri,
    )
    evaluate.set_display_name("6 - Évaluer sur test")

    # 7. Exécuter les contrôles Evidently et les tests comportementaux.
    # Cette étape peut s'exécuter en parallèle avec l'évaluation.
    model_check = validate_model(
        model=train.outputs["model"],
        train_data=prepare.outputs["train_data"],
        test_data=prepare.outputs["test_data"],
    )
    model_check.set_display_name("7 - Valider le modèle")

    # 8. Enregistrer uniquement après réussite des deux contrôles.
    register = register_model(
        run_id=train.outputs["Output"],
        mlflow_uri=mlflow_uri,
    )
    register.after(evaluate, model_check)
    register.set_display_name("8 - Enregistrer le modèle")

    # Téléchargement : tentatives supplémentaires en cas d'échec réseau.
    ingest.set_retry(
        num_retries=3,
        backoff_duration="30s",
    )
    ingest.set_cpu_limit("500m")
    ingest.set_memory_limit("512Mi")
    kubernetes.set_timeout(ingest, 600)

    # Ressources des étapes de données.
    validation.set_memory_limit("512Mi")
    prepare.set_memory_limit("1Gi")
    drift_check.set_memory_limit("1Gi")

    # Cache pour les premières étapes de données.
    for task in (ingest, validation, prepare):
        task.set_caching_options(True)

    # Entraînement : ressources et durée maximale.
    train.set_cpu_request("500m")
    train.set_cpu_limit("2")
    train.set_memory_request("512Mi")
    train.set_memory_limit("2Gi")
    kubernetes.set_timeout(train, 1800)

    # Ressources des contrôles et de l'enregistrement.
    evaluate.set_memory_limit("1Gi")
    model_check.set_memory_limit("2Gi")
    register.set_memory_limit("1Gi")

    # Réexécuter les contrôles et produire un nouvel entraînement.
    for task in (
        drift_check,
        train,
        evaluate,
        model_check,
        register,
    ):
        task.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=scoring_pipeline,
        package_path="scoring_pipeline.yaml",
    )
    print("Pipeline compilé : scoring_pipeline.yaml")
