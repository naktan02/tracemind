"""Query adaptation dataset으로 supervised LoRA baseline을 실행하는 helper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from scripts.labeled_query_rows import LabeledQueryRow

from .query_adaptation_io import (
    QueryAdaptationLoraExportArtifacts,
    build_labeled_rows_from_query_adaptation_dataset,
    write_query_adaptation_lora_dataset,
)
from .runner import run_supervised_lora_baseline


@dataclass(slots=True)
class PreparedQueryAdaptationSupervisedRun:
    """query adaptation dataset을 baseline 실행 입력과 trace export로 정리한 결과."""

    cfg: DictConfig
    train_jsonl_ref: str
    eval_set_refs: dict[str, str]
    run_id: str
    export_dir: Path
    selection_example_count: int
    train_rows: list[LabeledQueryRow]
    eval_rows_by_name: dict[str, list[LabeledQueryRow]]
    train_artifacts: QueryAdaptationLoraExportArtifacts
    selection_set_name: str
    selection_artifacts: QueryAdaptationLoraExportArtifacts | None = None
    eval_artifacts: dict[str, QueryAdaptationLoraExportArtifacts] = field(
        default_factory=dict
    )

    @property
    def export_outputs(self) -> dict[str, str]:
        outputs = {
            "train_jsonl": str(self.train_artifacts.jsonl_path),
            "train_manifest": str(self.train_artifacts.manifest_path),
            "train_summary": str(self.train_artifacts.summary_path),
        }
        if self.selection_artifacts is not None:
            outputs[f"{self.selection_set_name}_jsonl"] = str(
                self.selection_artifacts.jsonl_path
            )
            outputs[f"{self.selection_set_name}_manifest"] = str(
                self.selection_artifacts.manifest_path
            )
            outputs[f"{self.selection_set_name}_summary"] = str(
                self.selection_artifacts.summary_path
            )
        for dataset_name, artifacts in self.eval_artifacts.items():
            outputs[f"{dataset_name}_jsonl"] = str(artifacts.jsonl_path)
            outputs[f"{dataset_name}_manifest"] = str(artifacts.manifest_path)
            outputs[f"{dataset_name}_summary"] = str(artifacts.summary_path)
        return outputs


def run_query_adaptation_supervised_baseline(
    *,
    cfg: DictConfig,
    train_dataset: Any,
    selection_dataset: Any | None = None,
    eval_datasets: Mapping[str, Any] | None = None,
    export_root: str | Path = "data/processed/query_adaptation_lora",
    selection_set_name: str = "selection",
    generated_at: datetime | None = None,
) -> dict[str, str]:
    """agent-local adaptation dataset으로 baseline을 실행하고 trace export를 남긴다."""

    prepared = prepare_query_adaptation_supervised_run(
        cfg=cfg,
        train_dataset=train_dataset,
        selection_dataset=selection_dataset,
        eval_datasets=eval_datasets,
        export_root=export_root,
        selection_set_name=selection_set_name,
        generated_at=generated_at,
    )
    run_outputs = run_supervised_lora_baseline(
        prepared.cfg,
        train_rows=prepared.train_rows,
        eval_rows_by_name=prepared.eval_rows_by_name,
        selection_set_name=prepared.selection_set_name,
        train_jsonl_ref=prepared.train_jsonl_ref,
        eval_set_refs=prepared.eval_set_refs,
    )
    return {
        **prepared.export_outputs,
        **run_outputs,
    }


def prepare_query_adaptation_supervised_run(
    *,
    cfg: DictConfig,
    train_dataset: Any,
    selection_dataset: Any | None = None,
    eval_datasets: Mapping[str, Any] | None = None,
    export_root: str | Path = "data/processed/query_adaptation_lora",
    selection_set_name: str = "selection",
    generated_at: datetime | None = None,
) -> PreparedQueryAdaptationSupervisedRun:
    """QueryAdaptationDataset를 baseline 실행 입력과 export path로 정규화한다."""

    effective_generated_at = generated_at or datetime.now(tz=timezone.utc)
    run_id = _resolve_query_adaptation_run_id(
        cfg=cfg,
        generated_at=effective_generated_at,
    )
    export_dir = Path(str(export_root)) / run_id
    export_dir.mkdir(parents=True, exist_ok=True)

    train_outputs = write_query_adaptation_lora_dataset(
        dataset=train_dataset,
        output_path=export_dir / "train.jsonl",
        annotation_source="query_adaptation_train",
        generated_at=effective_generated_at,
    )
    train_rows = build_labeled_rows_from_query_adaptation_dataset(
        train_dataset,
        annotation_source="query_adaptation_train",
    )

    eval_output_paths: dict[str, str] = {}
    eval_rows_by_name: dict[str, list[LabeledQueryRow]] = {}

    effective_selection_dataset = selection_dataset or train_dataset
    selection_artifacts: QueryAdaptationLoraExportArtifacts | None = None
    if selection_dataset is None:
        selection_path = str(train_outputs.jsonl_path)
        selection_rows = list(train_rows)
    else:
        selection_artifacts = write_query_adaptation_lora_dataset(
            dataset=selection_dataset,
            output_path=export_dir / f"{selection_set_name}.jsonl",
            annotation_source=f"query_adaptation_{selection_set_name}",
            generated_at=effective_generated_at,
        )
        selection_path = str(selection_artifacts.jsonl_path)
        selection_rows = build_labeled_rows_from_query_adaptation_dataset(
            selection_dataset,
            annotation_source=f"query_adaptation_{selection_set_name}",
        )
    eval_output_paths[selection_set_name] = selection_path
    eval_rows_by_name[selection_set_name] = selection_rows

    effective_eval_datasets = {} if eval_datasets is None else dict(eval_datasets)
    eval_artifacts: dict[str, QueryAdaptationLoraExportArtifacts] = {}
    for dataset_name, dataset in effective_eval_datasets.items():
        if dataset_name == selection_set_name and selection_dataset is None:
            continue
        eval_outputs = write_query_adaptation_lora_dataset(
            dataset=dataset,
            output_path=export_dir / f"{dataset_name}.jsonl",
            annotation_source=f"query_adaptation_{dataset_name}",
            generated_at=effective_generated_at,
        )
        eval_output_paths[dataset_name] = str(eval_outputs.jsonl_path)
        eval_artifacts[dataset_name] = eval_outputs
        eval_rows_by_name[dataset_name] = (
            build_labeled_rows_from_query_adaptation_dataset(
                dataset,
                annotation_source=f"query_adaptation_{dataset_name}",
            )
        )

    return PreparedQueryAdaptationSupervisedRun(
        cfg=cfg,
        train_jsonl_ref=str(train_outputs.jsonl_path),
        eval_set_refs=dict(eval_output_paths),
        run_id=run_id,
        export_dir=export_dir,
        selection_example_count=effective_selection_dataset.count,
        train_rows=train_rows,
        eval_rows_by_name=eval_rows_by_name,
        train_artifacts=train_outputs,
        selection_set_name=selection_set_name,
        selection_artifacts=selection_artifacts,
        eval_artifacts=eval_artifacts,
    )


def _resolve_query_adaptation_run_id(
    *,
    cfg: DictConfig,
    generated_at: datetime,
) -> str:
    trainer_version = str(getattr(cfg, "trainer_version", "") or "").strip()
    if trainer_version:
        return trainer_version
    return generated_at.strftime("query_adapt_lora_%Y_%m_%d_%H%M%S")


__all__ = [
    "PreparedQueryAdaptationSupervisedRun",
    "prepare_query_adaptation_supervised_run",
    "run_query_adaptation_supervised_baseline",
]
