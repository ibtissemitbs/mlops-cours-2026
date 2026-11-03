"""Construit une requête de prédiction depuis le CSV de test."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument(
        "--csv",
        default=str(ROOT / "data/processed/test.csv"),
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("-n doit être supérieur ou égal à 1")

    cfg = load_config(ROOT / "dataset.yaml")
    rows = pd.read_csv(args.csv)[cfg.features].head(args.n)
    rows = rows.astype(object).where(rows.notna(), None)

    print(
        json.dumps(
            {"instances": rows.to_dict("records")},
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
