"""Export SQLite experiment indexes into static dashboard JSON."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data/processed/experiment_index/experiment_results.sqlite")
DEFAULT_OUTPUT_PATH = Path("apps/experiment_dashboard/data/experiment_dashboard.json")


def build_dashboard_bundle(db_path: Path) -> dict[str, Any]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        runs = _fetch_rows(connection, "select * from experiment_runs order by run_id")
        eval_metrics = _fetch_rows(
            connection,
            "select * from eval_metrics order by run_id, eval_set",
        )
        per_class_metrics = _fetch_rows(
            connection,
            "select * from per_class_metrics order by run_id, eval_set, category",
        )
        confusion_matrix_cells = _fetch_rows(
            connection,
            """
            select * from confusion_matrix_cells
            order by run_id, eval_set, actual_category, predicted_category
            """,
        )
        epoch_metrics = _fetch_rows(
            connection,
            "select * from epoch_metrics order by run_id, epoch",
        )
        epoch_per_class_metrics = _fetch_rows(
            connection,
            "select * from epoch_per_class_metrics order by run_id, epoch, category",
        )

    return {
        "schema_version": "experiment_dashboard_data.v1",
        "filters": _build_filters(
            runs=runs,
            eval_metrics=eval_metrics,
            per_class_metrics=per_class_metrics,
        ),
        "runs": runs,
        "eval_metrics": eval_metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix_cells": confusion_matrix_cells,
        "epoch_metrics": epoch_metrics,
        "epoch_per_class_metrics": epoch_per_class_metrics,
    }


def write_dashboard_bundle(*, db_path: Path, output_path: Path) -> dict[str, Any]:
    bundle = build_dashboard_bundle(db_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export experiment result SQLite index to dashboard JSON.",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    bundle = write_dashboard_bundle(db_path=args.db, output_path=args.output)
    print(
        f"dashboard_json={args.output} "
        f"runs={len(bundle['runs'])} eval_metrics={len(bundle['eval_metrics'])}",
        flush=True,
    )


def _fetch_rows(connection: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query)]


def _build_filters(
    *,
    runs: list[dict[str, Any]],
    eval_metrics: list[dict[str, Any]],
    per_class_metrics: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    return {
        "tracks": _unique(row.get("track") for row in runs),
        "method_families": _unique(row.get("method_family") for row in runs),
        "methods": _unique(row.get("method_name") for row in runs),
        "algorithms": _unique(row.get("algorithm_name") for row in runs),
        "selection_slugs": _unique(row.get("selection_slug") for row in runs),
        "labeled_datasets": _unique(row.get("labeled_dataset_name") for row in runs),
        "unlabeled_datasets": _unique(
            row.get("unlabeled_dataset_name") for row in runs
        ),
        "validation_datasets": _unique(
            row.get("validation_dataset_name") for row in runs
        ),
        "test_datasets": _unique(row.get("test_dataset_name") for row in runs),
        "eval_sets": _unique(row.get("eval_set") for row in eval_metrics),
        "categories": _unique(row.get("category") for row in per_class_metrics),
        "learning_rates": _unique(row.get("learning_rate") for row in runs),
        "classifier_learning_rates": _unique(
            row.get("classifier_learning_rate") for row in runs
        ),
        "seeds": _unique(row.get("seed") for row in runs),
    }


def _unique(values) -> list[Any]:
    return sorted({value for value in values if value is not None})


if __name__ == "__main__":
    main()
