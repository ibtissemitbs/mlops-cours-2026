"""Télécharge AI4I ou lit une copie CSV locale.

Usage :
    python -m src.download
    python -m src.download --from-file /chemin/ai4i2020.csv
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.dataset_config import load_config

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
)


def from_uci() -> pd.DataFrame:
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as response:
        content = response.read()

    with (
        zipfile.ZipFile(io.BytesIO(content)) as archive,
        archive.open("ai4i2020.csv") as csv_file,
    ):
        return pd.read_csv(csv_file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-file", help="Chemin d'un CSV AI4I local")
    args = parser.parse_args()

    cfg = load_config()

    if cfg.name != "ai4i-predictive-maintenance":
        raise ValueError("Remplacez d'abord dataset.yaml par la configuration AI4I.")

    if args.from_file:
        df = pd.read_csv(args.from_file)
        source = "CSV local"
    else:
        df = from_uci()
        source = "UCI"

    cfg.validate_columns(df)

    for column in cfg.numeric:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[cfg.target].isna().any() or not df[cfg.target].isin([0, 1]).all():
        raise ValueError("La cible doit contenir uniquement 0 et 1.")

    out = Path(cfg.raw_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cfg.columns].to_csv(out, index=False)

    share = cfg.encode_target(df[cfg.target]).mean()
    print(f"{len(df)} lignes ({source}) -> {out} · classe '{cfg.positive_class}' : {share:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
