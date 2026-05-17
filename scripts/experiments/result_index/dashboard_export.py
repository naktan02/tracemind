"""Export SQLite experiment indexes into static dashboard JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data/processed/experiment_index/experiment_results.sqlite")
DEFAULT_OUTPUT_PATH = Path("apps/experiment_dashboard/data/experiment_dashboard.json")


def build_dashboard_bundle(
    db_path: Path,
    *,
    artifact_output_dir: Path | None = None,
) -> dict[str, Any]:
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
        artifacts = _fetch_rows(
            connection,
            "select * from artifacts order by run_id, eval_set, artifact_kind",
        )
    projection_images = _build_projection_images(
        artifacts=artifacts,
        artifact_output_dir=artifact_output_dir,
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
        "projection_images": projection_images,
    }


def write_dashboard_bundle(*, db_path: Path, output_path: Path) -> dict[str, Any]:
    artifact_output_dir = output_path.parent / "artifacts"
    bundle = build_dashboard_bundle(
        db_path,
        artifact_output_dir=artifact_output_dir,
    )
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


def _build_projection_images(
    *,
    artifacts: list[dict[str, Any]],
    artifact_output_dir: Path | None,
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.get("artifact_kind") != "projection_png":
            continue
        source_path = Path(str(artifact["artifact_ref"]))
        if not source_path.exists():
            continue
        run_id = str(artifact["run_id"])
        eval_set = str(artifact["eval_set"] or "unknown")
        image_src = str(source_path)
        if artifact_output_dir is not None:
            target_dir = artifact_output_dir / _safe_path_part(run_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{_safe_path_part(eval_set)}.projection.png"
            shutil.copy2(source_path, target_path)
            image_src = f"data/artifacts/{target_dir.name}/{target_path.name}"
        images.append(
            {
                "run_id": run_id,
                "eval_set": eval_set,
                "image_src": image_src,
                "reducer": artifact.get("reducer"),
                "fallback_reason": artifact.get("fallback_reason"),
            }
        )
    return images


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
        "client_counts": _unique(row.get("client_count") for row in runs),
        "round_budgets": _unique(row.get("round_budget") for row in runs),
        "shard_policies": _unique(row.get("shard_policy_name") for row in runs),
        "shard_alphas": _unique(row.get("shard_alpha") for row in runs),
        "adapter_families": _unique(row.get("adapter_family_name") for row in runs),
        "aggregation_backends": _unique(
            row.get("aggregation_backend_name") for row in runs
        ),
        "update_delta_formats": _unique(row.get("update_delta_format") for row in runs),
        "embedding_backends": _unique(row.get("embedding_backend") for row in runs),
        "embedding_model_ids": _unique(row.get("embedding_model_id") for row in runs),
        "embedding_devices": _unique(row.get("embedding_device") for row in runs),
        "local_trainer_devices": _unique(
            row.get("local_trainer_device") for row in runs
        ),
    }


def _unique(values) -> list[Any]:
    return sorted({value for value in values if value is not None})


def _safe_path_part(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


if __name__ == "__main__":
    main()
