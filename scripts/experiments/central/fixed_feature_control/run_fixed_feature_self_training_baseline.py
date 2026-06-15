"""중앙 fixed-feature self-training 준지도 baseline entrypoint."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import joblib
from omegaconf import DictConfig, OmegaConf

from methods.classification.fixed_feature.self_training import (
    FixedFeatureUnlabeledDataset,
    run_fixed_feature_self_training_classification,
)
from methods.classification.fixed_feature.training import FixedFeatureDataset
from scripts.experiments.central.fixed_feature_control import (
    run_fixed_feature_baseline as supervised_fixed_feature_runner,
)
from scripts.support.artifacts.run_artifacts import build_run_dir
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    get_labeled_query_row_mapped_label,
    load_labeled_query_rows,
)

FIXED_FEATURE_SELF_TRAINING_REPORT_SCHEMA_VERSION = (
    "central_fixed_feature_self_training_eval.v1"
)


@hydra.main(
    version_base=None,
    config_path="../../../../conf",
    config_name=(
        "entrypoints/central/fixed_feature_control/"
        "run_fixed_feature_self_training_baseline"
    ),
)
def main(cfg: DictConfig) -> None:
    outputs = run_fixed_feature_self_training_baseline(cfg)
    for key, value in outputs.items():
        print(f"{key}={value}")


def run_fixed_feature_self_training_baseline(cfg: DictConfig) -> dict[str, str]:
    """Hydra config가 고른 fixed-feature self-training baseline을 실행한다."""

    created_at = datetime.now(
        supervised_fixed_feature_runner.FIXED_FEATURE_RUN_TIMEZONE
    )
    feature_space_config = _plain_mapping(cfg.fixed_feature_space)
    estimator_config = _plain_mapping(cfg.fixed_feature_estimator)
    self_training_config = _plain_mapping(cfg.fixed_feature_self_training)
    categories = [str(category) for category in cfg.fixed_categories]
    trainer_version = _resolve_trainer_version(
        cfg=cfg,
        created_at=created_at,
        feature_space_name=str(feature_space_config["name"]),
        estimator_name=str(estimator_config["name"]),
    )

    train_rows = load_labeled_query_rows(Path(str(cfg.train_jsonl)))
    unlabeled_pool_rows = load_labeled_query_rows(Path(str(cfg.unlabeled_jsonl)))
    unlabeled_rows, unlabeled_sampling = _select_unlabeled_rows(
        rows=unlabeled_pool_rows,
        cfg=cfg,
        self_training_config=self_training_config,
    )
    eval_rows_by_name = {
        str(name): load_labeled_query_rows(Path(str(path)))
        for name, path in cfg.eval_sets.items()
    }
    if str(cfg.selection_set) not in eval_rows_by_name:
        raise ValueError(
            f"selection_set '{cfg.selection_set}' is not included in eval_sets."
        )

    result = run_fixed_feature_self_training_classification(
        feature_space_config=feature_space_config,
        estimator_config=estimator_config,
        self_training_config=self_training_config,
        categories=categories,
        train_dataset=_to_dataset(train_rows),
        unlabeled_dataset=_to_unlabeled_dataset(unlabeled_rows),
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
        unlabeled_pool_row_count=len(unlabeled_pool_rows),
        unlabeled_rows=unlabeled_rows,
        unlabeled_sampling=unlabeled_sampling,
        eval_rows_by_name=eval_rows_by_name,
        result=result,
        feature_space_config=feature_space_config,
        estimator_config=estimator_config,
        self_training_config=self_training_config,
    )


def _to_dataset(rows: list[LabeledQueryRow]) -> FixedFeatureDataset:
    return FixedFeatureDataset(
        texts=[str(row["text"]) for row in rows],
        labels=[get_labeled_query_row_mapped_label(row) for row in rows],
    )


def _to_unlabeled_dataset(rows: list[LabeledQueryRow]) -> FixedFeatureUnlabeledDataset:
    return FixedFeatureUnlabeledDataset(
        texts=[str(row["text"]) for row in rows],
        query_ids=[str(row["query_id"]) for row in rows],
    )


def _write_artifacts(
    *,
    cfg: DictConfig,
    run_dir: Path,
    trainer_version: str,
    created_at: datetime,
    categories: list[str],
    train_rows: list[LabeledQueryRow],
    unlabeled_pool_row_count: int,
    unlabeled_rows: list[LabeledQueryRow],
    unlabeled_sampling: Mapping[str, Any],
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]],
    result: Any,
    feature_space_config: Mapping[str, Any],
    estimator_config: Mapping[str, Any],
    self_training_config: Mapping[str, Any],
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
    pseudo_label_path = artifacts_dir / "pseudo_labels.train_unlabeled.jsonl"
    report_path = reports_dir / "report.json"
    log_path = logs_dir / "training_log.jsonl"
    joblib.dump(result.estimator, model_path)
    joblib.dump(result.feature_space, feature_space_path)
    _write_json(
        label_schema_path,
        {
            "categories": categories,
            "label_field": "mapped_label_4",
            "unlabeled_training_label": int(
                self_training_config.get("unlabeled_label", -1)
            ),
        },
    )
    _write_pseudo_labels(path=pseudo_label_path, result=result)

    prediction_paths: dict[str, str] = {}
    for dataset_name, rows in eval_rows_by_name.items():
        prediction_path = artifacts_dir / f"predictions.{dataset_name}.jsonl"
        supervised_fixed_feature_runner._write_predictions(
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
        unlabeled_row_count=len(unlabeled_rows),
        unlabeled_pool_row_count=unlabeled_pool_row_count,
        unlabeled_sampling=unlabeled_sampling,
        eval_rows_by_name=eval_rows_by_name,
        result=result,
        feature_space_config=feature_space_config,
        estimator_config=estimator_config,
        self_training_config=self_training_config,
        artifact_paths={
            "model": str(model_path),
            "feature_space": str(feature_space_path),
            "label_schema": str(label_schema_path),
            "pseudo_labels": str(pseudo_label_path),
            "predictions": prediction_paths,
        },
    )
    _write_json(report_path, report)
    _write_jsonl(
        log_path,
        [
            {
                "event": "fixed_feature_self_training_baseline_completed",
                "trainer_version": trainer_version,
                "feature_space": str(feature_space_config["name"]),
                "estimator": str(estimator_config["name"]),
                "labeled_rows": len(train_rows),
                "unlabeled_rows": len(unlabeled_rows),
                "unlabeled_pool_rows": unlabeled_pool_row_count,
                "accepted_pseudo_label_count": (
                    result.summary.accepted_pseudo_label_count
                ),
            }
        ],
    )
    return {
        "output_dir": str(run_dir),
        "model": str(model_path),
        "feature_space": str(feature_space_path),
        "label_schema": str(label_schema_path),
        "pseudo_labels": str(pseudo_label_path),
        "report_json": str(report_path),
    }


def _build_report(
    *,
    cfg: DictConfig,
    trainer_version: str,
    created_at: datetime,
    categories: list[str],
    train_row_count: int,
    unlabeled_row_count: int,
    unlabeled_pool_row_count: int,
    unlabeled_sampling: Mapping[str, Any],
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]],
    result: Any,
    feature_space_config: Mapping[str, Any],
    estimator_config: Mapping[str, Any],
    self_training_config: Mapping[str, Any],
    artifact_paths: Mapping[str, Any],
) -> dict[str, Any]:
    results = {
        dataset_name: evaluation.report
        for dataset_name, evaluation in result.evaluations.items()
    }
    return {
        "schema_version": FIXED_FEATURE_SELF_TRAINING_REPORT_SCHEMA_VERSION,
        "trainer_version": trainer_version,
        "created_at": created_at.isoformat(),
        "manifest": {
            "trainer_version": trainer_version,
            "track": "central_ssl_fixed_feature_self_training",
            "feature_space": dict(feature_space_config),
            "estimator": dict(estimator_config),
            "self_training": dict(self_training_config),
            "unlabeled_sampling": dict(unlabeled_sampling),
            "self_training_summary": asdict(result.summary),
            "categories": categories,
            "selection_set": str(cfg.selection_set),
            "runtime": _plain_mapping(cfg.runtime),
            "paper_backbone": _plain_mapping(cfg.paper_backbone),
            "central_ssl_budget": _plain_mapping(cfg.central_ssl_budget),
            "train_batch_size": int(cfg.train_batch_size),
            "eval_batch_size": int(cfg.eval_batch_size),
            "query_labeled_budget": _plain_mapping(cfg.query_labeled_budget),
            "query_data_selection": _plain_mapping(cfg.query_data_selection),
            "query_data_selection_slug": str(cfg.query_data_selection_slug),
            "train_jsonl": str(cfg.train_jsonl),
            "unlabeled_jsonl": str(cfg.unlabeled_jsonl),
            "eval_sets": {str(name): str(path) for name, path in cfg.eval_sets.items()},
            "row_counts": {
                "train_labeled": train_row_count,
                "train_unlabeled_pool": unlabeled_pool_row_count,
                "train_unlabeled": unlabeled_row_count,
                "eval": {
                    dataset_name: len(rows)
                    for dataset_name, rows in eval_rows_by_name.items()
                },
            },
            "artifacts": dict(artifact_paths),
        },
        "results": results,
    }


def _select_unlabeled_rows(
    *,
    rows: list[LabeledQueryRow],
    cfg: DictConfig,
    self_training_config: Mapping[str, Any],
) -> tuple[list[LabeledQueryRow], dict[str, Any]]:
    cap_policy = str(self_training_config.get("unlabeled_cap_policy", "step_budget"))
    explicit_cap = self_training_config.get("max_unlabeled_rows")
    max_unlabeled_rows = (
        _resolve_step_budget_unlabeled_cap(cfg)
        if explicit_cap is None and cap_policy == "step_budget"
        else None
        if explicit_cap is None
        else int(explicit_cap)
    )
    sample_seed = int(self_training_config.get("unlabeled_sample_seed", cfg.seed))
    if max_unlabeled_rows is None or max_unlabeled_rows >= len(rows):
        return list(rows), {
            "policy": cap_policy,
            "pool_count": len(rows),
            "used_count": len(rows),
            "max_unlabeled_rows": max_unlabeled_rows,
            "sample_seed": sample_seed,
            "sampled": False,
        }
    if max_unlabeled_rows <= 0:
        raise ValueError("fixed_feature_self_training.max_unlabeled_rows must be > 0.")
    sampled_rows = list(rows)
    random.Random(sample_seed).shuffle(sampled_rows)
    selected_rows = sampled_rows[:max_unlabeled_rows]
    return selected_rows, {
        "policy": cap_policy,
        "pool_count": len(rows),
        "used_count": len(selected_rows),
        "max_unlabeled_rows": max_unlabeled_rows,
        "sample_seed": sample_seed,
        "sampled": True,
    }


def _resolve_step_budget_unlabeled_cap(cfg: DictConfig) -> int | None:
    max_train_steps = getattr(cfg, "max_train_steps", None)
    if max_train_steps is None:
        max_train_steps = getattr(
            getattr(cfg, "central_ssl_budget", None), "max_train_steps", None
        )
    if max_train_steps is None:
        return None
    return int(max_train_steps) * int(cfg.train_batch_size)


def _write_pseudo_labels(*, path: Path, result: Any) -> None:
    _write_jsonl(path, [asdict(record) for record in result.pseudo_label_records])


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
        "fixed_feature_self_training_"
        f"{feature_space_name}_{estimator_name}_%Y_%m_%d_%H%M%S"
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
