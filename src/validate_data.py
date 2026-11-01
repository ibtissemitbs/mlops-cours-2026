"""Valide les données avec le contrat Great Expectations versionné dans gx/."""

import argparse
import html
import json
from pathlib import Path

import great_expectations as gx
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

from src.dataset_config import DatasetConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE = "scoring_contract"
CHECKPOINT = "scoring_gate"
REFERENCE_CSV = Path("data/processed/train.csv")
REPORT_JSON = Path("reports/validation.json")
REPORT_HTML = Path("reports/validation.html")

MOSTLY = 0.99
NUMERIC_TYPES = ["int64", "int32", "float64", "float32", "Int64", "Float64"]
TEXT_TYPES = ["object", "str", "string", "category"]


def build_expectations(cfg: DatasetConfig, ref: pd.DataFrame) -> list:
    """Construit les règles depuis le contrat et les données d'entraînement."""
    cfg.validate_columns(ref)

    E = gx.expectations
    expectations = []
    add = expectations.append

    numeric = [
        col
        for col in cfg.columns
        if col != cfg.target and pd.api.types.is_numeric_dtype(ref[col])
    ]
    categorical = [
        col for col in cfg.columns
        if col not in numeric and col != cfg.target
    ]

    # Structure du fichier.
    add(E.ExpectTableColumnsToMatchSet(
        column_set=cfg.columns,
        exact_match=False,
    ))
    add(E.ExpectTableRowCountToBeBetween(
        min_value=max(1, len(ref) // 10),
    ))

    # Valeurs manquantes et types.
    for col in cfg.columns:
        null_rate = float(ref[col].isna().mean())
        options = (
            {"mostly": round(max(0.0, 1 - 2 * null_rate), 3)}
            if null_rate else {}
        )
        add(E.ExpectColumnValuesToNotBeNull(column=col, **options))

        types = (
            NUMERIC_TYPES
            if pd.api.types.is_numeric_dtype(ref[col])
            else TEXT_TYPES
        )
        add(E.ExpectColumnValuesToBeInTypeList(
            column=col,
            type_list=types,
        ))

    # Bornes numériques larges pour détecter les erreurs grossières.
    for col in numeric:
        lo = float(ref[col].min())
        hi = float(ref[col].max())
        span = hi - lo

        add(E.ExpectColumnValuesToBeBetween(
            column=col,
            min_value=0.0 if lo >= 0 else lo - span,
            max_value=hi + span,
            mostly=MOSTLY,
        ))

    # Catégories observées à l'entraînement.
    for col in categorical:
        observed = sorted(ref[col].dropna().unique().tolist(), key=str)
        add(E.ExpectColumnValuesToBeInSet(
            column=col,
            value_set=observed,
            mostly=MOSTLY,
        ))

    # Classes de la cible et proportion de la classe positive.
    classes = sorted(ref[cfg.target].dropna().unique().tolist(), key=str)
    positive = [
        value for value in classes
        if str(value) == str(cfg.positive_class)
    ]
    others = [
        value for value in classes
        if str(value) != str(cfg.positive_class)
    ]
    rate = float(cfg.encode_target(ref[cfg.target]).mean())

    add(E.ExpectColumnValuesToBeInSet(
        column=cfg.target,
        value_set=classes,
    ))
    add(E.ExpectColumnValuesToBeInSet(
        column=cfg.target,
        value_set=positive,
        mostly=round(rate / 2, 3),
        description=f"Au moins {rate / 2:.2%} de classe positive",
    ))
    add(E.ExpectColumnValuesToBeInSet(
        column=cfg.target,
        value_set=others,
        mostly=round(max(0.0, 1 - 2 * rate), 3),
        description=f"Au plus {min(1.0, 2 * rate):.2%} de classe positive",
    ))

    return expectations


def save_suite(expectations: list) -> gx.ExpectationSuite:
    """Enregistre la suite, la source et le checkpoint dans gx/."""
    context = gx.get_context(
        mode="file",
        project_root_dir=PROJECT_ROOT,
    )
    context.enable_analytics(False)
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    context.variables.save()

    batch = (
        context.data_sources.add_or_update_pandas("scoring")
        .add_dataframe_asset("dataframe")
        .add_batch_definition_whole_dataframe("full")
    )

    suite = context.suites.add_or_update(
        gx.ExpectationSuite(name=SUITE, expectations=expectations)
    )
    definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=SUITE, data=batch, suite=suite)
    )

    context.checkpoints.add_or_update(
        gx.Checkpoint(
            name=CHECKPOINT,
            validation_definitions=[definition],
            actions=[
                gx.checkpoint.UpdateDataDocsAction(name="data_docs")
            ],
            result_format={"result_format": "SUMMARY"},
        )
    )
    return suite


def validate(cfg: DatasetConfig, df: pd.DataFrame):
    """Exécute le contrat enregistré sur le DataFrame fourni."""
    checkpoint = PROJECT_ROOT / "gx" / "checkpoints" / f"{CHECKPOINT}.json"
    if not checkpoint.exists():
        raise SystemExit(
            "Suite absente : lancez python -m src.validate_data --build-suite"
        )

    context = gx.get_context(
        mode="file",
        project_root_dir=PROJECT_ROOT,
    )
    result = context.checkpoints.get(CHECKPOINT).run(
        batch_parameters={"dataframe": df}
    )
    return next(iter(result.run_results.values()))


def check_name(result) -> str:
    config = result.expectation_config
    column = config.kwargs.get("column")
    name = config.type + (f"({column})" if column else "")
    return f"{name} : {config.description}" if config.description else name


def run_checks(cfg: DatasetConfig, df: pd.DataFrame) -> dict[str, bool]:
    """Conserve l'interface utilisée au Lab 2."""
    return {
        check_name(result): bool(result.success)
        for result in validate(cfg, df).results
    }


def write_html(results, summary: dict) -> None:
    """Produit un rapport HTML autonome pour Kubeflow."""
    rows = []

    for result in sorted(results, key=lambda item: item.success):
        parameters = {
            key: value
            for key, value in result.expectation_config.kwargs.items()
            if key not in ("column", "batch_id")
        }
        status = "OK" if result.success else "ÉCHEC"
        color = "#1a7f37" if result.success else "#cf222e"
        details = result.result or {}
        exception = result.exception_info or {}
        if exception.get("raised_exception"):
            details = {"erreur": exception.get("exception_message")}

        rows.append(
            "<tr>"
            f"<td>{html.escape(check_name(result))}</td>"
            f"<td>{html.escape(str(parameters))}</td>"
            f"<td>{html.escape(str(details))}</td>"
            f"<td style='color:{color}'>{status}</td>"
            "</tr>"
        )

    REPORT_HTML.write_text(
        "<!DOCTYPE html><html lang='fr'><head>"
        "<meta charset='utf-8'><title>Validation des données</title>"
        "<style>body{font-family:sans-serif;margin:2em}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccc;padding:8px;text-align:left}"
        "td{overflow-wrap:anywhere}</style></head><body>"
        f"<h1>{SUITE}</h1>"
        f"<p>{summary['n_failed']} échec(s) sur {summary['n_checks']} contrôles"
        f" — {summary['rows']} lignes</p>"
        "<table><tr><th>Règle</th><th>Attendu</th>"
        "<th>Observé</th><th>Statut</th></tr>"
        + "".join(rows)
        + "</table></body></html>",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", help="Fichier CSV à vérifier")
    parser.add_argument(
        "--build-suite",
        action="store_true",
        help="Génère le contrat depuis les données d'entraînement",
    )
    args = parser.parse_args()
    cfg = load_config()

    if args.build_suite:
        ref = pd.read_csv(args.csv or REFERENCE_CSV)
        suite = save_suite(build_expectations(cfg, ref))
        print(
            f"Suite {SUITE} : {len(suite.expectations)} expectations "
            "-> gx/expectations/"
        )
        return 0

    df = pd.read_csv(args.csv or cfg.raw_path)
    results = validate(cfg, df).results
    failed = [check_name(result) for result in results if not result.success]
    has_target = cfg.target in df.columns and len(df) > 0

    summary = {
        "success": not failed,
        "n_checks": len(results),
        "n_failed": len(failed),
        "failed": failed,
        "rows": len(df),
        "positive_rate": (
            round(float(cfg.encode_target(df[cfg.target]).mean()), 4)
            if has_target else None
        ),
        "suite": SUITE,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = json.dumps(summary, indent=2, ensure_ascii=False)
    REPORT_JSON.write_text(report + "\n", encoding="utf-8")
    write_html(results, summary)
    print(report)

    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
