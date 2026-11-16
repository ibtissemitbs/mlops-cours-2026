"""Vérifie les manifestes GitOps du Lab 6 sans contacter le cluster."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from pipelines import gitops

ROOT = Path(__file__).resolve().parents[1]


def render(env, root=ROOT):
    folder = str(root / "k8s" / "overlays" / env)
    if shutil.which("kustomize"):
        command = ["kustomize", "build", folder]
    elif shutil.which("kubectl"):
        command = ["kubectl", "kustomize", folder]
    else:
        pytest.fail("Installer kustomize ou kubectl pour vérifier le rendu des manifestes.")

    result = subprocess.run(command, check=True, capture_output=True, text=True)
    documents = [doc for doc in yaml.safe_load_all(result.stdout) if doc]
    assert len(documents) == 1
    return documents[0]


@pytest.mark.parametrize("env", ["staging", "prod"])
def test_overlay(env):
    service = render(env)
    assert service["kind"] == "InferenceService"
    assert service["metadata"]["name"] == "scoring-model"
    assert service["metadata"]["namespace"] == gitops.NAMESPACES[env]
    assert service["metadata"]["annotations"]["serving.kserve.io/deploymentMode"] == "Knative"

    predictor = service["spec"]["predictor"]
    name, tag = gitops.read_image(ROOT, env)
    assert predictor["containers"][0]["image"] == f"{name}:{tag}"

    if env == "staging":
        assert (predictor["minReplicas"], predictor["maxReplicas"]) == (0, 1)
        assert "canaryTrafficPercent" not in predictor
    else:
        assert (predictor["minReplicas"], predictor["maxReplicas"]) == (1, 5)
        assert predictor["canaryTrafficPercent"] == gitops.read_canary(ROOT)
        assert 0 <= predictor["canaryTrafficPercent"] <= 100


def test_applications_argocd():
    path = ROOT / "k8s" / "argocd-app.yaml"
    apps = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    assert sorted(app["metadata"]["name"] for app in apps) == [
        "scoring-prod",
        "scoring-staging",
    ]

    for app in apps:
        assert app["apiVersion"] == "argoproj.io/v1alpha1"
        assert app["kind"] == "Application"
        assert app["metadata"]["namespace"] == "argocd"
        env = app["metadata"]["name"].removeprefix("scoring-")
        spec = app["spec"]
        assert spec["source"]["targetRevision"] == "main"
        assert spec["source"]["path"] == f"k8s/overlays/{env}"
        assert spec["destination"]["namespace"] == gitops.NAMESPACES[env]
        assert spec["syncPolicy"]["automated"] == {"prune": True, "selfHeal": True}


def test_gitops_modifie_image_et_canary(tmp_path):
    shutil.copytree(ROOT / "k8s", tmp_path / "k8s")
    image = "ghcr.io/ibtissemitbs/mlops-cours-2026/scoring-api"
    gitops.set_image(tmp_path, "prod", image, "2.1.7")
    gitops.set_canary(tmp_path, 10)

    predictor = render("prod", tmp_path)["spec"]["predictor"]
    assert predictor["containers"][0]["image"] == f"{image}:2.1.7"
    assert predictor["canaryTrafficPercent"] == 10


def test_gitops_refuse_un_canary_invalide(tmp_path):
    shutil.copytree(ROOT / "k8s", tmp_path / "k8s")
    with pytest.raises(SystemExit):
        gitops.set_canary(tmp_path, 120)
    assert gitops.read_canary(tmp_path) == gitops.read_canary(ROOT)
