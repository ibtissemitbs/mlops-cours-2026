"""Tests du contrat FastAPI et de l'export ONNX du projet."""

import json
import re
from pathlib import Path

import onnxruntime as ort
import pytest

from pipelines.gitops import payload

ROOT = Path(__file__).resolve().parents[1]


def test_health(api):
    response = api.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_metadata_et_version(api, contract):
    response = api.get("/metadata")
    assert response.status_code == 200
    metadata = response.json()
    assert metadata["api_version"] == "9.9.9"
    assert metadata["threshold"] == 0.4
    assert metadata["features"] == {
        "numeric": contract.numeric,
        "categorical": contract.categorical,
    }

    version = api.get("/version")
    assert version.status_code == 200
    assert version.json() == {"version": "9.9.9", "model_version": "test"}


def test_predict_coherent_avec_le_seuil(api, contract):
    response = api.post("/predict", json=payload(n=3))
    assert response.status_code == 200
    body = response.json()
    assert body["positive_class"] == str(contract.positive_class)
    assert body["threshold"] == 0.4
    assert len(body["predictions"]) == 3

    for result in body["predictions"]:
        probability = result["probability"]
        assert 0.0 <= probability <= 1.0
        assert result["prediction"] == int(probability >= body["threshold"])


def test_valeur_manquante_acceptee(api, contract):
    row = payload()["instances"][0] | {contract.features[0]: None}
    response = api.post("/predict", json={"instances": [row]})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "case", ["colonne manquante", "mauvais type", "colonne en trop", "liste vide", "objet nu"]
)
def test_requete_invalide_refusee(api, contract, case):
    row = payload()["instances"][0]
    numeric_column = contract.numeric[0]
    bodies = {
        "colonne manquante": {
            "instances": [{key: value for key, value in row.items() if key != numeric_column}]
        },
        "mauvais type": {"instances": [row | {numeric_column: "12"}]},
        "colonne en trop": {"instances": [row | {"colonne_inconnue": 1}]},
        "liste vide": {"instances": []},
        "objet nu": row,
    }
    assert api.post("/predict", json=bodies[case]).status_code == 422


def test_modele_versionne_conforme_au_contrat(contract):
    folder = ROOT / "packaging" / "model"
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["features"] == {
        "numeric": contract.numeric,
        "categorical": contract.categorical,
    }
    assert metadata["positive_class"] == str(contract.positive_class)
    assert 0.0 < metadata["threshold"] < 1.0

    session = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
    columns = [*contract.numeric, *contract.categorical]
    expected_names = [re.sub(r"[^a-zA-Z0-9_]", "_", column) for column in columns]
    assert [entry.name for entry in session.get_inputs()] == expected_names
    assert "probabilities" in [output.name for output in session.get_outputs()]
