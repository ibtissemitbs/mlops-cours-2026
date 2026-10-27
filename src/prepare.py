"""Prépare les données d'entraînement, de test et la source offline Feast."""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.dataset_config import load_config

PROCESSED = Path("data/processed")


def main() -> None:
    cfg = load_config()
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["prepare"]
    id_col = params["id_column"]

    df = pd.read_csv(cfg.raw_path)
    cfg.validate_columns(df)

    # Identifiant reproductible pour chaque observation.
    if id_col not in df.columns:
        df.insert(0, id_col, range(1, len(df) + 1))

    df = df[[id_col, *cfg.columns]]

    # Conserve une proportion similaire de pannes dans les deux ensembles.
    y = cfg.encode_target(df[cfg.target])
    train, test = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # Validation issue uniquement des données d'entraînement.
    train, validation = train_test_split(
        train,
        test_size=params["validation_size"],
        random_state=cfg.random_state,
        stratify=cfg.encode_target(train[cfg.target]),
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)
    validation.to_csv(PROCESSED / "validation.csv", index=False)
    print(
        f"Validation : {len(validation)} observations · "
        f"pannes : {cfg.encode_target(validation[cfg.target]).mean():.2%}"
    )

    # Source Feast : variables explicatives, sans la cible.
    features = df[[id_col, *cfg.features]].copy()
    features[cfg.numeric] = features[cfg.numeric].astype("float64")
    features["event_timestamp"] = pd.Timestamp(params["event_timestamp"])
    features.to_parquet(PROCESSED / "features.parquet", index=False)

    train_rate = cfg.encode_target(train[cfg.target]).mean()
    test_rate = cfg.encode_target(test[cfg.target]).mean()
    print(
        f"Train : {len(train)} / Test : {len(test)} · "
        f"classe '{cfg.positive_class}' : "
        f"{train_rate:.2%} / {test_rate:.2%}"
    )


if __name__ == "__main__":
    main()
