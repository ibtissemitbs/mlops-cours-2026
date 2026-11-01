"""Simule des températures plus élevées et détecte le drift."""

from pathlib import Path

import yaml
from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Output

from pipelines.components import (
    PIPELINE_IMAGE,
    detect_drift,
    ingest_data,
    prepare_features,
)

ROOT = Path(__file__).resolve().parents[1]
DRIFT = yaml.safe_load(
    (ROOT / "params.yaml").read_text(encoding="utf-8")
)["drift"]


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def simulate_drift(
    data: Input[Dataset],
    drifted: Output[Dataset],
) -> None:
    """Crée une copie modifiée selon params.yaml."""
    from pipelines.stage import run_stage

    clean = "data/drift/clean.csv"
    shifted = "data/drift/drifted.csv"

    run_stage(
        "src.simulate",
        args=["drift", clean, shifted],
        inputs={clean: data.path},
        outputs={shifted: drifted.path},
    )


@dsl.pipeline(
    name="scoring-drift-demo",
    description="Détecter un drift simulé sur les températures AI4I.",
)
def drift_pipeline(
    ks_alpha: float = float(DRIFT["ks_alpha"]),
    psi_threshold: float = float(DRIFT["psi_threshold"]),
    drift_share: float = float(DRIFT["drift_share"]),
    fail_on_drift: bool = False,
):
    ingest = ingest_data()
    ingest.set_display_name("1 - Télécharger AI4I")

    prepare = prepare_features(
        raw_data=ingest.outputs["raw_data"],
    )
    prepare.set_display_name("2 - Préparer les données")

    shifted = simulate_drift(
        data=prepare.outputs["test_data"],
    )
    shifted.set_display_name("3 - Simuler des températures plus élevées")

    detection = detect_drift(
        reference_data=prepare.outputs["train_data"],
        current_data=shifted.outputs["drifted"],
        ks_alpha=ks_alpha,
        psi_threshold=psi_threshold,
        drift_share=drift_share,
        fail_on_drift=fail_on_drift,
    )
    detection.set_display_name("4 - Détecter le drift simulé")

    ingest.set_memory_limit("512Mi")
    prepare.set_memory_limit("1Gi")
    shifted.set_memory_limit("512Mi")
    detection.set_memory_limit("1Gi")

    for task in (ingest, prepare, shifted, detection):
        task.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=drift_pipeline,
        package_path="drift_demo.yaml",
    )
    print("Pipeline compilé : drift_demo.yaml")

