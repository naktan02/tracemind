"""Teacher bootstrap seed/unlabeled split 준비 helper."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from scripts.datasets.lib.split import split_rows
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
    load_labeled_query_rows,
)

BOOTSTRAP_SPLIT_SCHEMA_VERSION = "fixed_classifier_teacher_split.v1"


@dataclass(slots=True)
class BootstrapSplitArtifacts:
    """teacher seed / unlabeled split 산출물."""

    seed_train_jsonl: Path
    unlabeled_jsonl: Path
    manifest_path: Path


@dataclass(slots=True)
class ResolvedTeacherRows:
    """teacher bootstrap에 사용할 seed/unlabeled row와 참조 경로."""

    seed_rows: list[LabeledQueryRow]
    unlabeled_rows: list[LabeledQueryRow]
    seed_jsonl_ref: str
    unlabeled_jsonl_ref: str
    split_artifacts: BootstrapSplitArtifacts | None


def resolve_teacher_and_unlabeled_rows(
    *,
    cfg: DictConfig,
    run_id: str,
    export_dir: Path,
    teacher_seed_rows: Sequence[LabeledQueryRow] | None,
    teacher_unlabeled_rows: Sequence[LabeledQueryRow] | None,
) -> ResolvedTeacherRows:
    """teacher seed rows와 unlabeled pool을 실행 입력으로 정규화한다."""

    if teacher_seed_rows is not None and teacher_unlabeled_rows is not None:
        return ResolvedTeacherRows(
            seed_rows=list(teacher_seed_rows),
            unlabeled_rows=list(teacher_unlabeled_rows),
            seed_jsonl_ref="in_memory/teacher_seed_rows.jsonl",
            unlabeled_jsonl_ref="in_memory/teacher_unlabeled_rows.jsonl",
            split_artifacts=None,
        )

    teacher_train_path = Path(str(cfg.teacher_train_jsonl))
    explicit_unlabeled = getattr(cfg, "teacher_unlabeled_jsonl", None)
    if teacher_seed_rows is None:
        base_rows = load_labeled_query_rows(teacher_train_path)
    else:
        base_rows = list(teacher_seed_rows)

    if teacher_unlabeled_rows is not None:
        return ResolvedTeacherRows(
            seed_rows=base_rows,
            unlabeled_rows=list(teacher_unlabeled_rows),
            seed_jsonl_ref=str(teacher_train_path),
            unlabeled_jsonl_ref="in_memory/teacher_unlabeled_rows.jsonl",
            split_artifacts=None,
        )

    if explicit_unlabeled:
        unlabeled_path = Path(str(explicit_unlabeled))
        return ResolvedTeacherRows(
            seed_rows=base_rows,
            unlabeled_rows=load_labeled_query_rows(unlabeled_path),
            seed_jsonl_ref=str(teacher_train_path),
            unlabeled_jsonl_ref=str(unlabeled_path),
            split_artifacts=None,
        )

    if not bool(cfg.bootstrap_split.enabled):
        raise ValueError(
            "Provide teacher_unlabeled_jsonl or enable bootstrap_split.enabled."
        )

    seed_rows, unlabeled_rows = split_rows(
        base_rows,
        validation_ratio=float(cfg.bootstrap_split.unlabeled_ratio),
        seed=int(cfg.bootstrap_split.seed),
    )
    seed_path = export_dir / "teacher_seed_train.jsonl"
    unlabeled_path = export_dir / "teacher_unlabeled_pool.jsonl"
    manifest_path = export_dir / "teacher_split.manifest.json"
    dump_labeled_query_rows(seed_path, seed_rows)
    dump_labeled_query_rows(unlabeled_path, unlabeled_rows)
    _write_json(
        manifest_path,
        {
            "schema_version": BOOTSTRAP_SPLIT_SCHEMA_VERSION,
            "bootstrap_version": run_id,
            "source_train_jsonl": str(teacher_train_path),
            "seed": int(cfg.bootstrap_split.seed),
            "unlabeled_ratio": float(cfg.bootstrap_split.unlabeled_ratio),
            "teacher_seed_row_count": len(seed_rows),
            "teacher_unlabeled_row_count": len(unlabeled_rows),
            "teacher_seed_jsonl": str(seed_path),
            "teacher_unlabeled_jsonl": str(unlabeled_path),
        },
    )
    return ResolvedTeacherRows(
        seed_rows=seed_rows,
        unlabeled_rows=unlabeled_rows,
        seed_jsonl_ref=str(seed_path),
        unlabeled_jsonl_ref=str(unlabeled_path),
        split_artifacts=BootstrapSplitArtifacts(
            seed_train_jsonl=seed_path,
            unlabeled_jsonl=unlabeled_path,
            manifest_path=manifest_path,
        ),
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
