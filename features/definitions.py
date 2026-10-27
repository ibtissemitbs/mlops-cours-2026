"""Définitions Feast construites depuis le contrat des données."""

import sys
from datetime import timedelta
from pathlib import Path

import yaml
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    ValueType,
)
from feast.types import Float64, String

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_config import load_config  # noqa: E402 — nécessite le chemin ROOT

cfg = load_config(ROOT / "dataset.yaml")
params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
id_column = params["prepare"]["id_column"]

project = Project(
    name="scoring",
    description=f"Variables du projet AI4I : {cfg.name}",
)

request = Entity(
    name="request",
    join_keys=[id_column],
    value_type=ValueType.INT64,
    description="Une observation industrielle identifiée par request_id",
)

request_source = FileSource(
    name="request_source",
    path=str(ROOT / "data" / "processed" / "features.parquet"),
    timestamp_field="event_timestamp",
)

request_features = FeatureView(
    name="request_features",
    entities=[request],
    ttl=timedelta(days=3650),
    schema=(
        [Field(name=c, dtype=Float64) for c in cfg.numeric]
        + [Field(name=c, dtype=String) for c in cfg.categorical]
    ),
    source=request_source,
    online=True,
    tags={"dataset": cfg.name, "owner": "mlops-cours"},
)

scoring_v1 = FeatureService(
    name="scoring_v1",
    features=[request_features],
)
