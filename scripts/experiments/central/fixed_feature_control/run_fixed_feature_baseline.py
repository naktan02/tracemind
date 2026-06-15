"""중앙 fixed-feature 지도학습 baseline entrypoint."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import hydra
import joblib
from omegaconf import DictConfig, OmegaConf

from methods.classification.fixed_feature.training import (
    FixedFeatureDataset,
    run_fixed_feature_classification,
)
from scripts.support.artifacts.run_artifacts import build_run_dir
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    get_labeled_query_row_mapped_label,
    load_labeled_query_rows,
)

FIXED_FEATURE_REPORT_SCHEMA_VERSION = "central_fixed_feature_classification_eval.v1"
FIXED_FEATURE_RUN_TIMEZONE = ZoneInfo("Asia/Seoul")


@hydra.main(
    version_base=None,
    config_path="../../../../conf",
    config_name="entrypoints/central/fixed_feature_control/run_fixed_feature_baseline",
)
def main(cfg: DictConfig) -> None:
    outputs = run_fixed_feature_baseline(cfg)
    for key, value in outputs.items():
        print(f"{key}={value}")


def run_fixed_feature_baseline(cfg: DictConfig) -> dict[str, str]:
    """Hydra config가 고른 fixed-feature baseline을 실행하고 산출물을 저장한다."""

    created_at = datetime.now(FIXED_FEATURE_RUN_TIMEZONE)
    feature_space_config = _plain_mapping(cfg.fixed_feature_space)
    estimator_config = _plain_mapping(cfg.fixed_feature_estimator)
    categories = [str(category) for category in cfg.fixed_categories]
    trainer_version = _resolve_trainer_version(
        cfg=cfg,
        created_at=created_at,
        feature_space_name=str(feature_space_config["name"]),
        estimator_name=str(estimator_config["name"]),
    )

    train_rows = load_labeled_query_rows(Path(str(cfg.train_jsonl)))
    eval_rows_by_name = {
        str(name): load_labeled_query_rows(Path(str(path)))
        for name, path in cfg.eval_sets.items()
    }
    if str(cfg.selection_set) not in eval_rows_by_name:
        raise ValueError(
            f"selection_set '{cfg.selection_set}' is not included in eval_sets."
        )

    result = run_fixed_feature_classification(
        feature_space_config=feature_space_config,
        estimator_config=estimator_config,
        categories=categories,
        train_dataset=_to_dataset(train_rows),
        eval_datasets={
            dataset_name: _to_dataset(rows)
            for dataset_name, rows in eval_rows_by_name.items()
        },
    )
    run_dir = build_run_dir(
        Path(str(cfg.output_dir)),
        run_id=trainer_version,
        created_at=created_at,
    )
    return _write_artifacts(
        cfg=cfg,
        run_dir=run_dir,
        trainer_version=trainer_version,
        created_at=created_at,
        categories=categories,
        train_rows=train_rows,
        eval_rows_by_name=eval_rows_by_name,
        result=result,
        feature_space_config=feature_space_config,
        estimator_config=estimator_config,
    )


def _to_dataset(rows: list[LabeledQueryRow]) -> FixedFeatureDataset:
    return FixedFeatureDataset(
        texts=[str(row["text"]) for row in rows],
        labels=[get_labeled_query_row_mapped_label(row) for row in rows],
    )


def _write_artifacts(
    *,
    cfg: DictConfig,
    run_dir: Path,
    trainer_version: str,
    created_at: datetime,
    categories: list[str],
    train_rows: list[LabeledQueryRow],
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]],
    result: Any,
    feature_space_config: Mapping[str, Any],
    estimator_config: Mapping[str, Any],
) -> dict[str, str]:
    artifacts_dir = run_dir / "artifacts"
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.joblib"
    feature_space_path = artifacts_dir / "feature_space.joblib"
    label_schema_path = artifacts_dir / "label_schema.json"
    report_path = reports_dir / "report.json"
    log_path = logs_dir / "training_log.jsonl"
    joblib.dump(result.estimator, model_path)
    joblib.dump(result.feature_space, feature_space_path)
    _write_json(
        label_schema_path,
        {
            "categories": categories,
            "label_field": "mapped_label_4",
        },
    )

    prediction_paths: dict[str, str] = {}
    for dataset_name, rows in eval_rows_by_name.items():
        prediction_path = artifacts_dir / f"predictions.{dataset_name}.jsonl"
        _write_predictions(
            path=prediction_path,
            rows=rows,
            categories=categories,
            predicted_labels=result.evaluations[dataset_name].predicted_labels,
            score_matrix=result.evaluations[dataset_name].score_matrix,
        )
        prediction_paths[dataset_name] = str(prediction_path)

    report = _build_report(
        cfg=cfg,
        trainer_version=trainer_version,
        created_at=created_at,
        categories=categories,
        train_row_count=len(train_rows),
        eval_rows_by_name=eval_rows_by_name,
        result=result,
        feature_space_config=feature_space_config,
        estimator_config=estimator_config,
        artifact_paths={
            "model": str(model_path),
            "feature_space": str(feature_space_path),
            "label_schema": str(label_schema_path),
            "predictions": prediction_paths,
        },
    )
    _write_json(report_path, report)
    _write_jsonl(
        log_path,
        [
            {
                "event": "fixed_feature_baseline_completed",
                "trainer_version": trainer_version,
                "feature_space": str(feature_space_config["name"]),
                "estimator": str(estimator_config["name"]),
                "train_rows": len(train_rows),
            }
        ],
    )
    return {
        "output_dir": str(run_dir),
        "model": str(model_path),
        "feature_space": str(feature_space_path),
        "label_schema": str(label_schema_path),
        "report_json": str(report_path),
    }


def _build_report(
    *,
    cfg: DictConfig,
    trainer_version: str,
    created_at: datetime,
    categories: list[str],
    train_row_count: int,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]],
    result: Any,
    feature_space_config: Mapping[str, Any],
    estimator_config: Mapping[str, Any],
    artifact_paths: Mapping[str, Any],
) -> dict[str, Any]:
    results = {
        dataset_name: evaluation.report
        for dataset_name, evaluation in result.evaluations.items()
    }
    return {
        "schema_version": FIXED_FEATURE_REPORT_SCHEMA_VERSION,
        "trainer_version": trainer_version,
        "created_at": created_at.isoformat(),
        "manifest": {
            "trainer_version": trainer_version,
            "track": "central_supervised_fixed_feature",
            "feature_space": dict(feature_space_config),
            "estimator": dict(estimator_config),
            "categories": categories,
            "selection_set": str(cfg.selection_set),
            "query_labeled_budget": _plain_mapping(cfg.query_labeled_budget),
            "query_data_selection": _plain_mapping(cfg.query_data_selection),
            "query_data_selection_slug": str(cfg.query_data_selection_slug),
            "train_jsonl": str(cfg.train_jsonl),
            "eval_sets": {str(name): str(path) for name, path in cfg.eval_sets.items()},
            "row_counts": {
                "train": train_row_count,
                "eval": {
                    dataset_name: len(rows)
                    for dataset_name, rows in eval_rows_by_name.items()
                },
            },
            "artifacts": dict(artifact_paths),
        },
        "results": results,
    }


def _write_predictions(
    *,
    path: Path,
    rows: list[LabeledQueryRow],
    categories: list[str],
    predicted_labels: list[str],
    score_matrix: Any,
) -> None:
    records = []
    for row, predicted_label, scores in zip(
        rows,
        predicted_labels,
        score_matrix,
        strict=True,
    ):
        records.append(
            {
                "query_id": str(row["query_id"]),
                "actual_label": get_labeled_query_row_mapped_label(row),
                "predicted_label": predicted_label,
                "score_by_category": {
                    category: round(float(score), 6)
                    for category, score in zip(categories, scores, strict=True)
                },
            }
        )
    _write_jsonl(path, records)


def _resolve_trainer_version(
    *,
    cfg: DictConfig,
    created_at: datetime,
    feature_space_name: str,
    estimator_name: str,
) -> str:
    configured = str(getattr(cfg, "trainer_version", "") or "").strip()
    if configured:
        return configured
    return created_at.strftime(
        f"fixed_feature_{feature_space_name}_{estimator_name}_%Y_%m_%d_%H%M%S"
    )


def _plain_mapping(source: Any) -> dict[str, Any]:
    raw = OmegaConf.to_container(source, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Expected config section to resolve to a dict.")
    return dict(raw)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
