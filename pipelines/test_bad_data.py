"""Vérifie que les données corrompues bloquent le pipeline."""

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Output

from pipelines.components import (
    MLFLOW_URI,
    PIPELINE_IMAGE,
    evaluate_model,
    ingest_data,
    prepare_features,
    register_model,
    train_model,
    validate_data,
)
from pipelines.scoring_pipeline import PARAMS


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def corrupt_data(
    raw_data: Input[Dataset],
    corrupted_data: Output[Dataset],
) -> None:
    """Introduit volontairement trois erreurs dans une copie."""
    from pipelines.stage import run_stage

    run_stage(
        "src.simulate",
        args=[
            "corrupt",
            "data/input.csv",
            "data/corrupted.csv",
        ],
        inputs={"data/input.csv": raw_data.path},
        outputs={"data/corrupted.csv": corrupted_data.path},
    )


@dsl.pipeline(
    name="scoring-bad-data-test",
    description="La validation GX doit refuser les données corrompues.",
)
def bad_data_pipeline():
    ingest = ingest_data()
    ingest.set_display_name("1 - Télécharger AI4I")

    corrupt = corrupt_data(
        raw_data=ingest.outputs["raw_data"],
    )
    corrupt.set_display_name("2 - Introduire les trois erreurs")

    validation = validate_data(
        raw_data=corrupt.outputs["corrupted_data"],
    )
    validation.set_display_name("3 - Validation GX : échec attendu")

    prepare = prepare_features(
        raw_data=corrupt.outputs["corrupted_data"],
    )
    prepare.after(validation)
    prepare.set_display_name("4 - Préparation : doit être bloquée")

    train = train_model(
        raw_data=corrupt.outputs["corrupted_data"],
        train_data=prepare.outputs["train_data"],
        validation_data=prepare.outputs["validation_data"],
        model_type=PARAMS["model"],
        C=float(PARAMS["C"]),
        class_weight=str(PARAMS["class_weight"]),
        n_estimators=int(PARAMS["n_estimators"]),
        max_depth=str(PARAMS["max_depth"]),
        learning_rate=float(PARAMS["learning_rate"]),
        random_state=int(PARAMS["random_state"]),
        hidden_layer_sizes=",".join(
            str(n) for n in PARAMS["hidden_layer_sizes"]
        ),
        alpha=float(PARAMS["alpha"]),
        learning_rate_init=float(PARAMS["learning_rate_init"]),
        max_iter=int(PARAMS["max_iter"]),
        mlflow_uri=MLFLOW_URI,
        kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    )
    train.set_display_name("5 - Entraînement : doit être bloqué")

    evaluate = evaluate_model(
        model=train.outputs["model"],
        test_data=prepare.outputs["test_data"],
        run_id=train.outputs["Output"],
        mlflow_uri=MLFLOW_URI,
    )
    evaluate.set_display_name("6 - Évaluation : doit être bloquée")

    register = register_model(
        run_id=train.outputs["Output"],
        mlflow_uri=MLFLOW_URI,
    )
    register.after(evaluate)
    register.set_display_name("7 - Enregistrement : doit être bloqué")

    ingest.set_memory_limit("512Mi")
    corrupt.set_memory_limit("512Mi")
    validation.set_memory_limit("512Mi")
    prepare.set_memory_limit("1Gi")
    train.set_memory_limit("2Gi")
    train.set_cpu_request("500m")
    train.set_cpu_limit("2")
    evaluate.set_memory_limit("1Gi")
    register.set_memory_limit("1Gi")

    for task in (
        ingest,
        corrupt,
        validation,
        prepare,
        train,
        evaluate,
        register,
    ):
        task.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=bad_data_pipeline,
        package_path="test_bad_data.yaml",
    )
    print("Pipeline compilé : test_bad_data.yaml")
