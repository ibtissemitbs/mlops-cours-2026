"""Fixtures du Lab 6 pour tester l'API avec un mini-modèle ONNX."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from skl2onnx import to_onnx
from skl2onnx.common.data_types import DoubleTensorType, StringTensorType
from skl2onnx.sklapi import CastTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.dataset_config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def contract():
    return load_config(ROOT / "dataset.yaml")


def synthetic_frame(cfg, n=300, seed=0):
    rng = np.random.default_rng(seed)
    data = {column: rng.normal(50, 10, n) for column in cfg.numeric}
    data.update({column: rng.choice(["a", "b", "c"], n) for column in cfg.categorical})
    frame = pd.DataFrame(data)
    signal = frame[cfg.numeric[0]] if cfg.numeric else rng.normal(50, 10, n)
    frame[cfg.target] = np.where(signal > 50, str(cfg.positive_class), "autre")
    return frame


@pytest.fixture(scope="session")
def make_frame():
    return synthetic_frame


@pytest.fixture(scope="session")
def mini_model(contract, tmp_path_factory):
    X, y = contract.split_xy(synthetic_frame(contract))
    numeric = Pipeline(
        [
            ("scale", StandardScaler()),
            ("float32", CastTransformer(dtype=np.float32)),
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(missing_values="", strategy="constant", fill_value="?")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, contract.numeric),
            ("cat", categorical, contract.categorical),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression()),
        ]
    ).fit(X, y)

    input_types = [(column, DoubleTensorType([None, 1])) for column in contract.numeric]
    input_types += [(column, StringTensorType([None, 1])) for column in contract.categorical]
    onnx_model = to_onnx(
        model, initial_types=input_types, target_opset=17, options={"zipmap": False}
    )

    folder = tmp_path_factory.mktemp("model")
    model_path = folder / "model.onnx"
    model_path.write_bytes(onnx_model.SerializeToString())
    (folder / "metadata.json").write_text(
        '{"model_name":"scoring-model","model_version":"test","threshold":0.4}',
        encoding="utf-8",
    )
    return model_path


@pytest.fixture(scope="session")
def api(mini_model):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MODEL_PATH", str(mini_model))
        patch.setenv("DATASET_CONFIG", str(ROOT / "dataset.yaml"))
        patch.setenv("APP_VERSION", "9.9.9")
        patch.delenv("DECISION_THRESHOLD", raising=False)
        patch.delitem(sys.modules, "app.serve", raising=False)

        try:
            serve = importlib.import_module("app.serve")
            with TestClient(serve.app) as client:
                yield client
        finally:
            sys.modules.pop("app.serve", None)
