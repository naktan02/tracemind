"""Pseudo-label self-training LoRA runner."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from scripts.experiments.query_lora_ssl.config.pseudo_label_algorithm import (
    build_pseudo_label_algorithm_manifest,
)
from scripts.experiments.query_lora_ssl.io.query_adaptation import (
    build_labeled_rows_from_query_adaptation_dataset,
)
from scripts.experiments.query_lora_ssl.runners.supervised import (
    run_supervised_lora_baseline,
)
from scripts.io.labeled_query_rows import (
    LabeledQueryRow,
    dump_labeled_query_rows,
    load_labeled_query_rows,
)

LABELED_ROW_EXPORT_SCHEMA_VERSION = "labeled_query_row_export.v1"
LABELED_ROW_SUMMARY_SCHEMA_VERSION = "labeled_query_row_summary.v1"


@dataclass(slots=True)
class LabeledRowExportArtifacts:
    """Labeled row JSONL export 산출물."""

    jsonl_path: Path
    manifest_path: Path
    summary_path: Path


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
    run_outputs = run_supervised_lora_baseline(
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

    effective_include_seed_rows = (
        bool(getattr(cfg, "include_seed_train_rows", True))
        if include_seed_train_rows is None
        else bool(include_seed_train_rows)
    )
    if effective_include_seed_rows:
        effective_seed_train_rows = (
            load_labeled_query_rows(
                Path(
                    str(cfg.train_jsonl if train_jsonl_ref is None else train_jsonl_ref)
                )
            )
            if seed_train_rows is None
            else list(seed_train_rows)
        )
    else:
        effective_seed_train_rows = []
    effective_pseudo_label_rows = _resolve_pseudo_label_rows(
        cfg=cfg,
        pseudo_label_jsonl=pseudo_label_jsonl,
        pseudo_label_rows=pseudo_label_rows,
        pseudo_label_dataset=pseudo_label_dataset,
    )
    if not effective_pseudo_label_rows:
        raise ValueError("pseudo_label_rows must not be empty.")

    _ensure_unique_query_ids(
        effective_seed_train_rows,
        item_name="seed_train_rows",
    )
    _ensure_unique_query_ids(
        effective_pseudo_label_rows,
        item_name="pseudo_label_rows",
    )
    overlapping_query_ids = sorted(
        {
            str(row["query_id"])
            for row in effective_seed_train_rows
            if str(row["query_id"])
            in {str(item["query_id"]) for item in effective_pseudo_label_rows}
        }
    )
    if overlapping_query_ids:
        raise ValueError(
            "seed_train_rows and pseudo_label_rows must use disjoint query_id values. "
            f"Found overlaps: {overlapping_query_ids[:5]}."
        )

    combined_train_rows = [
        *effective_seed_train_rows,
        *effective_pseudo_label_rows,
    ]
    pseudo_label_artifacts = _write_labeled_row_export(
        rows=effective_pseudo_label_rows,
        output_path=export_dir / "pseudo_label_train.jsonl",
        generated_at=effective_generated_at,
    )
    combined_train_artifacts = _write_labeled_row_export(
        rows=combined_train_rows,
        output_path=export_dir / "combined_train.jsonl",
        generated_at=effective_generated_at,
    )

    return PreparedPseudoLabelSelfTrainingRun(
        cfg=cfg,
        train_jsonl_ref=str(combined_train_artifacts.jsonl_path),
        run_id=run_id,
        export_dir=export_dir,
        seed_train_count=len(effective_seed_train_rows),
        pseudo_label_count=len(effective_pseudo_label_rows),
        combined_train_count=len(combined_train_rows),
        combined_train_rows=combined_train_rows,
        pseudo_label_rows=effective_pseudo_label_rows,
        combined_train_artifacts=combined_train_artifacts,
        pseudo_label_artifacts=pseudo_label_artifacts,
    )


def _resolve_pseudo_label_rows(
    *,
    cfg: DictConfig,
    pseudo_label_jsonl: str | Path | None,
    pseudo_label_rows: Sequence[LabeledQueryRow] | None,
    pseudo_label_dataset: Any | None,
) -> list[LabeledQueryRow]:
    provided_sources = sum(
        source is not None
        for source in (pseudo_label_jsonl, pseudo_label_rows, pseudo_label_dataset)
    )
    if provided_sources > 1:
        raise ValueError(
            "Provide only one of pseudo_label_jsonl, pseudo_label_rows, or "
            "pseudo_label_dataset."
        )
    if pseudo_label_dataset is not None:
        return build_labeled_rows_from_query_adaptation_dataset(
            pseudo_label_dataset,
            annotation_source="pseudo_label_self_training",
        )
    if pseudo_label_rows is not None:
        return list(pseudo_label_rows)

    effective_path = pseudo_label_jsonl or getattr(cfg, "pseudo_label_jsonl", None)
    if effective_path is None:
        raise ValueError(
            "pseudo_label_jsonl is required when pseudo_label_rows/dataset is not "
            "provided."
        )
    return load_labeled_query_rows(Path(str(effective_path)))


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
    return generated_at.strftime("lora_pseudo_label_%Y_%m_%d_%H%M%S")


def _write_labeled_row_export(
    *,
    rows: Sequence[LabeledQueryRow],
    output_path: str | Path,
    generated_at: datetime,
) -> LabeledRowExportArtifacts:
    resolved_output_path = Path(str(output_path))
    dump_labeled_query_rows(resolved_output_path, rows)
    manifest_path = resolved_output_path.with_suffix(
        f"{resolved_output_path.suffix}.manifest.json"
    )
    summary_path = resolved_output_path.with_suffix(
        f"{resolved_output_path.suffix}.summary.json"
    )
    manifest = {
        "schema_version": LABELED_ROW_EXPORT_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "row_count": len(rows),
        "label_counts": dict(
            sorted(Counter(str(row["mapped_label_4"]) for row in rows).items())
        ),
        "raw_label_scheme_counts": dict(
            sorted(Counter(str(row["raw_label_scheme"]) for row in rows).items())
        ),
    }
    summary = {
        "schema_version": LABELED_ROW_SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "row_count": len(rows),
        "unique_query_id_count": len({str(row["query_id"]) for row in rows}),
        "annotation_source_counts": dict(
            sorted(Counter(str(row["annotation_source"]) for row in rows).items())
        ),
        "approved_by_counts": dict(
            sorted(
                Counter(
                    "none" if row["approved_by"] is None else str(row["approved_by"])
                    for row in rows
                ).items()
            )
        ),
        "locale_counts": dict(
            sorted(Counter(str(row["locale"]) for row in rows).items())
        ),
        "label_counts": manifest["label_counts"],
        "raw_label_scheme_counts": manifest["raw_label_scheme_counts"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return LabeledRowExportArtifacts(
        jsonl_path=resolved_output_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )


def _ensure_unique_query_ids(
    rows: Sequence[LabeledQueryRow],
    *,
    item_name: str,
) -> None:
    counts = Counter(str(row["query_id"]) for row in rows)
    duplicates = sorted(query_id for query_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{item_name} must not contain duplicate query_id values: {duplicates[:5]}."
        )
