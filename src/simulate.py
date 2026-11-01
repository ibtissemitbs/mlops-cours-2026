"""Crée des copies de test : données corrompues ou drift simulé."""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.dataset_config import DatasetConfig, load_config


def corrupt(cfg: DatasetConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Introduit trois erreurs dans une copie des données."""
    out = df.copy()
    seed = cfg.random_state

    # Mauvaise unité sur la première variable numérique.
    out[cfg.numeric[0]] = out[cfg.numeric[0]] * 100

    # Catégorie inconnue sur 10 % des lignes.
    indices = out.sample(frac=0.10, random_state=seed).index
    out.loc[indices, cfg.categorical[0]] = "INCONNU"

    # Cible manquante sur 5 % des lignes.
    indices = out.sample(frac=0.05, random_state=seed + 1).index
    out.loc[indices, cfg.target] = None

    return out


def drift(df: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    """Déplace les valeurs selon le scénario défini dans params.yaml."""
    out = df.copy()

    for col, factor in (scenario.get("scale") or {}).items():
        out[col] = out[col] * factor

    for col, delta in (scenario.get("shift") or {}).items():
        out[col] = (out[col] + delta).clip(df[col].min(), df[col].max())

    for col in out.columns:
        if (
            pd.api.types.is_integer_dtype(df[col])
            and not pd.api.types.is_integer_dtype(out[col])
        ):
            out[col] = out[col].round().astype(df[col].dtype)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["corrupt", "drift"])
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    cfg = load_config()
    df = pd.read_csv(args.input)

    if args.kind == "corrupt":
        out = corrupt(cfg, df)
    else:
        params = yaml.safe_load(
            Path("params.yaml").read_text(encoding="utf-8")
        )
        out = drift(df, params["drift"]["simulation"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"{args.kind} : {len(out)} lignes -> {args.output}")


if __name__ == "__main__":
    main()
