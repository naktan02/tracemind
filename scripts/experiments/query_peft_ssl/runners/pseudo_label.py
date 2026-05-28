"""Pseudo-label self-training PEFT runner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from scripts.experiments.query_peft_ssl.config.pseudo_label_algorithm import (
    build_pseudo_label_algorithm_manifest,
)
from scripts.experiments.query_peft_ssl.io.labeled_row_export import (
    LabeledRowExportArtifacts,
    write_labeled_row_export,
)
from scripts.experiments.query_peft_ssl.runners.pseudo_label_inputs import (
    resolve_pseudo_label_training_rows,
)
from scripts.experiments.query_peft_ssl.runners.supervised import (
    run_supervised_peft_baseline,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


@dataclass(slots=True)
class PreparedPseudoLabelSelfTrainingRun:
    """Pseudo-label self-training 실행 전 정규화 결과."""

    cfg: DictConfig
    train_jsonl_ref: str
    run_id: str
    export_dir: Path
    seed_train_count: int
    pseudo_label_count: int
    combined_train_count: int
    combined_train_rows: list[LabeledQueryRow]
    pseudo_label_rows: list[LabeledQueryRow]
    combined_train_artifacts: LabeledRowExportArtifacts
    pseudo_label_artifacts: LabeledRowExportArtifacts

    @property
    def export_outputs(self) -> dict[str, str]:
        return {
            "combined_train_jsonl": str(self.combined_train_artifacts.jsonl_path),
            "combined_train_manifest": str(self.combined_train_artifacts.manifest_path),
            "combined_train_summary": str(self.combined_train_artifacts.summary_path),
            "pseudo_label_jsonl": str(self.pseudo_label_artifacts.jsonl_path),
            "pseudo_label_manifest": str(self.pseudo_label_artifacts.manifest_path),
            "pseudo_label_summary": str(self.pseudo_label_artifacts.summary_path),
        }

    @property
    def manifest_overrides(self) -> dict[str, Any]:
        manifest = {
            "ssl_input_mode": "pseudo_label_replay",
            "pseudo_label_replay": {
                "training_mode": "supervised_replay",
                "label_source": "offline_pseudo_label_artifact",
            },
            "pseudo_label_jsonl": str(self.pseudo_label_artifacts.jsonl_path),
            "pseudo_label_manifest": str(self.pseudo_label_artifacts.manifest_path),
            "pseudo_label_summary": str(self.pseudo_label_artifacts.summary_path),
            "pseudo_label_row_count": self.pseudo_label_count,
            "seed_train_row_count": self.seed_train_count,
            "include_seed_train_rows": self.seed_train_count > 0,
            "combined_train_jsonl": str(self.combined_train_artifacts.jsonl_path),
            "combined_train_manifest": str(self.combined_train_artifacts.manifest_path),
            "combined_train_summary": str(self.combined_train_artifacts.summary_path),
            "combined_train_row_count": self.combined_train_count,
        }
        pseudo_label_algorithm_manifest = build_pseudo_label_algorithm_manifest(
            self.cfg
        )
        if pseudo_label_algorithm_manifest is not None:
            manifest["pseudo_label_algorithm"] = pseudo_label_algorithm_manifest
        return manifest


def run_pseudo_label_self_training(
    *,
    cfg: DictConfig,
    pseudo_label_jsonl: str | Path | None = None,
    pseudo_label_rows: Sequence[LabeledQueryRow] | None = None,
    pseudo_label_dataset: Any | None = None,
    seed_train_rows: Sequence[LabeledQueryRow] | None = None,
    include_seed_train_rows: bool | None = None,
    train_jsonl_ref: str | Path | None = None,
    trainer_version_override: str | None = None,
    export_root: str | Path | None = None,
    generated_at: datetime | None = None,
    categories_override: Sequence[str] | None = None,
) -> dict[str, str]:
    """Pseudo-labeled rows를 중심으로 self-training을 실행한다.

    canonical 경로는 새 accepted row만 사용한다. seed replay는 ablation/helper로만
    opt-in 한다.
    """

    prepared = prepare_pseudo_label_self_training_run(
        cfg=cfg,
        pseudo_label_jsonl=pseudo_label_jsonl,
        pseudo_label_rows=pseudo_label_rows,
        pseudo_label_dataset=pseudo_label_dataset,
        seed_train_rows=seed_train_rows,
        include_seed_train_rows=include_seed_train_rows,
        train_jsonl_ref=train_jsonl_ref,
        trainer_version_override=trainer_version_override,
        export_root=export_root,
        generated_at=generated_at,
    )
    run_outputs = run_supervised_peft_baseline(
        prepared.cfg,
        train_rows=prepared.combined_train_rows,
        train_jsonl_ref=prepared.train_jsonl_ref,
        trainer_version_override=prepared.run_id,
        extra_manifest=prepared.manifest_overrides,
        categories_override=(
            None
            if categories_override is None
            else [str(category) for category in categories_override]
        ),
    )
    return {
        **prepared.export_outputs,
        **run_outputs,
    }


def prepare_pseudo_label_self_training_run(
    *,
    cfg: DictConfig,
    pseudo_label_jsonl: str | Path | None = None,
    pseudo_label_rows: Sequence[LabeledQueryRow] | None = None,
    pseudo_label_dataset: Any | None = None,
    seed_train_rows: Sequence[LabeledQueryRow] | None = None,
    include_seed_train_rows: bool | None = None,
    train_jsonl_ref: str | Path | None = None,
    trainer_version_override: str | None = None,
    export_root: str | Path | None = None,
    generated_at: datetime | None = None,
) -> PreparedPseudoLabelSelfTrainingRun:
    """Pseudo-label self-training 입력을 baseline runner 형식으로 정규화한다."""

    effective_generated_at = generated_at or datetime.now(tz=timezone.utc)
    run_id = _resolve_run_id(
        cfg=cfg,
        generated_at=effective_generated_at,
        trainer_version_override=trainer_version_override,
    )
    resolved_export_root = (
        Path(str(cfg.pseudo_label_export_root))
        if export_root is None
        else Path(str(export_root))
    )
    export_dir = resolved_export_root / run_id
    export_dir.mkdir(parents=True, exist_ok=True)

    resolved_rows = resolve_pseudo_label_training_rows(
        cfg=cfg,
        pseudo_label_jsonl=pseudo_label_jsonl,
        pseudo_label_rows=pseudo_label_rows,
        pseudo_label_dataset=pseudo_label_dataset,
        seed_train_rows=seed_train_rows,
        include_seed_train_rows=include_seed_train_rows,
        train_jsonl_ref=train_jsonl_ref,
    )
    pseudo_label_artifacts = write_labeled_row_export(
        rows=resolved_rows.pseudo_label_rows,
        output_path=export_dir / "pseudo_label_train.jsonl",
        generated_at=effective_generated_at,
    )
    combined_train_artifacts = write_labeled_row_export(
        rows=resolved_rows.combined_train_rows,
        output_path=export_dir / "combined_train.jsonl",
        generated_at=effective_generated_at,
    )

    return PreparedPseudoLabelSelfTrainingRun(
        cfg=cfg,
        train_jsonl_ref=str(combined_train_artifacts.jsonl_path),
        run_id=run_id,
        export_dir=export_dir,
        seed_train_count=len(resolved_rows.seed_train_rows),
        pseudo_label_count=len(resolved_rows.pseudo_label_rows),
        combined_train_count=len(resolved_rows.combined_train_rows),
        combined_train_rows=resolved_rows.combined_train_rows,
        pseudo_label_rows=resolved_rows.pseudo_label_rows,
        combined_train_artifacts=combined_train_artifacts,
        pseudo_label_artifacts=pseudo_label_artifacts,
    )


def _resolve_run_id(
    *,
    cfg: DictConfig,
    generated_at: datetime,
    trainer_version_override: str | None = None,
) -> str:
    if trainer_version_override:
        return str(trainer_version_override).strip()
    trainer_version = str(getattr(cfg, "trainer_version", "") or "").strip()
    if trainer_version:
        return trainer_version
    return generated_at.strftime("query_peft_pseudo_label_%Y_%m_%d_%H%M%S")
