"""Tests comportementaux du modèle de maintenance prédictive."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.dataset_config import load_config
from src.train import MODEL_DIR, TEST_CSV, compute_metrics

CFG = load_config()
EXPECT = yaml.safe_load(
    Path(__file__).with_name("behavior.yaml").read_text(encoding="utf-8")
)

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "MLmodel").exists() or not TEST_CSV.exists(),
    reason="Modèle local ou données de test absents.",
)


@pytest.fixture(scope="module")
def model():
    import mlflow.sklearn
    from mlflow.models import Model

    metadata = Model.load(str(MODEL_DIR)).metadata
    threshold = float(metadata["decision_threshold"])
    estimator = mlflow.sklearn.load_model(str(MODEL_DIR))
    return estimator, threshold


@pytest.fixture(scope="module")
def data():
    raw = pd.read_csv(TEST_CSV)
    X, y = CFG.split_xy(raw)
    X[CFG.numeric] = X[CFG.numeric].astype("float64")
    return X, y, raw


def risk(model, X):
    """Probabilité de panne : le modèle est entraîné sur la cible encodée."""
    estimator, _ = model
    classes = list(estimator.classes_)
    positive_index = classes.index(1)
    return estimator.predict_proba(X)[:, positive_index]


def test_metrique_primaire_au_seuil(model, data):
    X, y, _ = data
    metrics = compute_metrics(CFG, y, risk(model, X), model[1])
    value = metrics[CFG.primary_metric]

    assert value >= CFG.promotion_threshold, (
        f"{CFG.primary_metric} = {value:.3f}, "
        f"minimum attendu = {CFG.promotion_threshold:.3f}"
    )


def test_cout_metier_inferieur_aux_politiques_naives(model, data):
    X, y, _ = data
    predictions = (risk(model, X) >= model[1]).astype(int)
    cost = CFG.business_cost(y, predictions)

    no_failure = np.zeros(len(y), dtype=int)
    all_failure = np.ones(len(y), dtype=int)
    best_naive_cost = min(
        CFG.business_cost(y, no_failure),
        CFG.business_cost(y, all_failure),
    )

    assert cost < best_naive_cost, (
        f"Coût du modèle = {cost}, "
        f"coût de la meilleure politique naïve = {best_naive_cost}"
    )

@pytest.mark.parametrize(
    "case",
    EXPECT.get("invariance") or [],
    ids=lambda case: case["column"],
)
def test_invariance(model, data, case):
    """Vérifie la stabilité face à une petite perturbation de mesure."""
    X, _, _ = data
    col = case["column"]

    assert col in CFG.numeric, (
        f"Variable numérique du contrat attendue : {col}"
    )
    shift = float(case["shift"])
    assert shift > 0, "La perturbation doit être strictement positive."

    original_decisions = risk(model, X) >= model[1]

    for direction in (-1, 1):
        modified = X.copy()
        modified[col] = modified[col] + direction * shift

        modified_decisions = risk(model, modified) >= model[1]
        flip_rate = float(
            np.mean(original_decisions != modified_decisions)
        )

        assert flip_rate <= case["max_flip_rate"], (
            f"{col}, variation {direction * shift:+g} : "
            f"{flip_rate:.1%} des décisions changent ; "
            f"maximum autorisé = {case['max_flip_rate']:.1%}"
        )
@pytest.mark.parametrize(
    "case",
    EXPECT.get("directional") or [],
    ids=lambda case: case["column"],
)
def test_directionnel(model, data, case):
    X, _, _ = data
    col = case["column"]

    assert col in CFG.numeric, f"Variable numérique attendue : {col}"
    assert case["expect"] in ("increase", "decrease")

    modified = X.copy()
    modified[col] = modified[col] * case["factor"]

    delta = risk(model, modified) - risk(model, X)
    sign = 1 if case["expect"] == "increase" else -1

    violation_rate = float(
        np.mean(sign * delta < -EXPECT["tolerance"])
    )

    assert violation_rate <= EXPECT["max_violation_rate"], (
        f"{col} : {violation_rate:.1%} des lignes violent l'attente ; "
        f"maximum autorisé = {EXPECT['max_violation_rate']:.1%}"
    )


@pytest.mark.parametrize(
    "attribute",
    CFG.sensitive,
    ids=lambda attribute: attribute.column,
)
def test_performance_par_sous_groupe(model, data, attribute):
    X, y, raw = data
    probabilities = risk(model, X)
    groups = attribute.groups(raw).to_numpy()
    scores = {}

    for group in np.unique(groups):
        mask = groups == group
        if (
            int(mask.sum()) >= EXPECT["min_group_size"]
            and y[mask].nunique() == 2
        ):
            metrics = compute_metrics(
                CFG,
                y[mask],
                probabilities[mask],
                model[1],
            )
            scores[str(group)] = metrics[CFG.primary_metric]

    if not scores:
        pytest.skip("Aucun sous-groupe suffisamment grand avec deux classes.")

    below = {
        group: score
        for group, score in scores.items()
        if score < CFG.promotion_threshold
    }
    assert not below, f"Sous-groupes sous le seuil : {below}"
