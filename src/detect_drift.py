"""Détecte un drift et génère des rapports JSON, Markdown et HTML."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

from src.dataset_config import load_config

REFERENCE_CSV = Path("data/processed/train.csv")
CURRENT_CSV = Path("data/processed/test.csv")
REPORT_JSON = Path("reports/drift.json")
REPORT_MD = Path("reports/drift.md")
REPORT_HTML = Path("reports/drift.html")
BINS = 10


def psi(ref: pd.Series, cur: pd.Series) -> float:
    """Mesure l'importance du changement de distribution."""
    ref = ref.dropna()
    cur = cur.dropna()

    if ref.empty or cur.empty:
        raise ValueError("Impossible de comparer une colonne sans valeurs.")

    if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > BINS:
        edges = np.unique(
            np.quantile(ref, np.linspace(0, 1, BINS + 1))
        )
        edges[0] = -np.inf
        edges[-1] = np.inf

        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    else:
        values = sorted(set(ref) | set(cur), key=str)
        r = (
            ref.value_counts(normalize=True)
            .reindex(values, fill_value=0)
            .to_numpy()
        )
        c = (
            cur.value_counts(normalize=True)
            .reindex(values, fill_value=0)
            .to_numpy()
        )

    r = np.clip(r, 1e-4, None)
    c = np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def column_drift(ref, cur, numeric: bool) -> dict:
    """Calcule les statistiques d'une colonne."""
    if numeric:
        clean_ref = ref.dropna()
        clean_cur = cur.dropna()

        if clean_ref.empty or clean_cur.empty:
            raise ValueError("Colonne numérique sans valeurs.")

        p_value = ks_2samp(clean_ref, clean_cur).pvalue
        scale = float(clean_ref.std())
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0

        distance = wasserstein_distance(clean_ref, clean_cur) / scale
        test = "KS"
    else:
        sample = np.repeat(
            ["reference", "courant"],
            [len(ref), len(cur)],
        )
        values = pd.concat([ref, cur]).astype(str).to_numpy()
        table = pd.crosstab(values, sample)

        p_value = chi2_contingency(table)[1] if len(table) > 1 else 1.0
        distance = None
        test = "chi2"

    return {
        "test": test,
        "p_value": float(p_value),
        "psi": psi(ref, cur),
        "wasserstein": None if distance is None else float(distance),
    }


def numeric_columns(cfg, ref) -> list:
    return [
        col
        for col in cfg.columns
        if col != cfg.target
        and col not in cfg.categorical
        and pd.api.types.is_numeric_dtype(ref[col])
    ]


def detect(cfg, ref, cur, ks_alpha, psi_threshold) -> list:
    """Décide, pour chaque colonne, si un drift est détecté."""
    cfg.validate_columns(ref)
    cfg.validate_columns(cur)

    numeric = numeric_columns(cfg, ref)
    rows = []

    for col in cfg.columns:
        stats = column_drift(ref[col], cur[col], col in numeric)
        drift = (
            stats["p_value"] < ks_alpha
            and stats["psi"] >= psi_threshold
        )
        rows.append({"column": col, **stats, "drift": bool(drift)})

    return rows


def evidently_report(cfg, ref, cur, drift_share, path) -> None:
    """Génère le rapport HTML Evidently."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    numeric = numeric_columns(cfg, ref)
    definition = DataDefinition(
        numerical_columns=numeric,
        categorical_columns=[
            col for col in cfg.columns if col not in numeric
        ],
    )

    report = Report([
        DataDriftPreset(
            columns=cfg.columns,
            drift_share=drift_share,
        )
    ])

    snapshot = report.run(
        Dataset.from_pandas(
            cur[cfg.columns],
            data_definition=definition,
        ),
        Dataset.from_pandas(
            ref[cfg.columns],
            data_definition=definition,
        ),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))


def markdown(rows, summary) -> str:
    verdict = "DRIFT" if summary["drift_detected"] else "pas de drift"
    lines = [
        f"### Drift : {summary['n_drifted']}/{summary['n_columns']} "
        f"colonnes ({verdict})",
        "",
        "| Colonne | Test | p-value | PSI | Wasserstein | Verdict |",
        "|---|---|---|---|---|---|",
    ]

    for row in sorted(rows, key=lambda item: -item["psi"]):
        distance = (
            ""
            if row["wasserstein"] is None
            else f"{row['wasserstein']:.3f}"
        )
        status = "DRIFT" if row["drift"] else "OK"
        lines.append(
            f"| {row['column']} | {row['test']} "
            f"| {row['p_value']:.4g} | {row['psi']:.3f} "
            f"| {distance} | {status} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    defaults = yaml.safe_load(
        Path("params.yaml").read_text(encoding="utf-8")
    )["drift"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference", nargs="?", default=str(REFERENCE_CSV)
    )
    parser.add_argument(
        "current", nargs="?", default=str(CURRENT_CSV)
    )
    parser.add_argument(
        "--ks-alpha", type=float, default=defaults["ks_alpha"]
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=defaults["psi_threshold"],
    )
    parser.add_argument(
        "--drift-share", type=float, default=defaults["drift_share"]
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Signale le drift sans retourner un code d'échec.",
    )
    args = parser.parse_args()

    cfg = load_config()
    ref = pd.read_csv(args.reference)
    cur = pd.read_csv(args.current)

    rows = detect(cfg, ref, cur, args.ks_alpha, args.psi_threshold)
    drifted = [row["column"] for row in rows if row["drift"]]
    share = len(drifted) / len(rows)

    summary = {
        "drift_detected": share >= args.drift_share,
        "n_columns": len(rows),
        "n_drifted": len(drifted),
        "share_drifted": round(share, 4),
        "drifted": drifted,
        "thresholds": {
            "ks_alpha": args.ks_alpha,
            "psi_threshold": args.psi_threshold,
            "drift_share": args.drift_share,
        },
        "rows": {"reference": len(ref), "current": len(cur)},
        "columns": rows,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    table = markdown(rows, summary)

    REPORT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(table, encoding="utf-8")
    evidently_report(cfg, ref, cur, args.drift_share, REPORT_HTML)

    print(table)
    print(f"Rapport HTML : {REPORT_HTML}")
    return 1 if summary["drift_detected"] and not args.no_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
