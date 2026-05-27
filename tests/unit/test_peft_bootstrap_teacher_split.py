from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.experiments.query_peft_ssl.runners.teacher_split import (
    resolve_teacher_and_unlabeled_rows,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
)


def _cfg(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "teacher_train_jsonl": str(tmp_path / "teacher_train.jsonl"),
            "teacher_unlabeled_jsonl": None,
            "bootstrap_split": {
                "enabled": False,
                "unlabeled_ratio": 0.5,
                "seed": 42,
            },
        }
    )


def _row(query_id: str, label: str, text: str) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=text,
        raw_label_scheme="manual_label",
        raw_label=label,
        mapped_label_4=label,
        locale="ko-KR",
        annotation_source="seed_train",
        approved_by="annotator",
        created_at="2026-04-14T00:00:00+00:00",
    )


def test_resolve_teacher_rows_keeps_in_memory_refs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    resolved = resolve_teacher_and_unlabeled_rows(
        cfg=cfg,
        run_id="bootstrap_v1",
        export_dir=tmp_path,
        teacher_seed_rows=[_row("s1", "anxiety", "seed")],
        teacher_unlabeled_rows=[_row("u1", "depression", "unlabeled")],
    )

    assert [row["query_id"] for row in resolved.seed_rows] == ["s1"]
    assert [row["query_id"] for row in resolved.unlabeled_rows] == ["u1"]
    assert resolved.seed_jsonl_ref == "in_memory/teacher_seed_rows.jsonl"
    assert resolved.unlabeled_jsonl_ref == "in_memory/teacher_unlabeled_rows.jsonl"
    assert resolved.split_artifacts is None


def test_resolve_teacher_rows_writes_split_manifest(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.bootstrap_split.enabled = True
    dump_labeled_query_rows(
        Path(str(cfg.teacher_train_jsonl)),
        [
            _row("q1", "anxiety", "a1"),
            _row("q2", "anxiety", "a2"),
            _row("q3", "depression", "d1"),
            _row("q4", "depression", "d2"),
        ],
    )

    resolved = resolve_teacher_and_unlabeled_rows(
        cfg=cfg,
        run_id="bootstrap_v1",
        export_dir=tmp_path / "exports",
        teacher_seed_rows=None,
        teacher_unlabeled_rows=None,
    )

    assert len(resolved.seed_rows) == 2
    assert len(resolved.unlabeled_rows) == 2
    assert resolved.split_artifacts is not None
    assert Path(resolved.seed_jsonl_ref).exists()
    assert Path(resolved.unlabeled_jsonl_ref).exists()

    manifest = json.loads(resolved.split_artifacts.manifest_path.read_text())
    assert manifest["schema_version"] == "fixed_classifier_teacher_split.v1"
    assert manifest["bootstrap_version"] == "bootstrap_v1"
    assert manifest["teacher_seed_row_count"] == 2
    assert manifest["teacher_unlabeled_row_count"] == 2


def test_resolve_teacher_rows_requires_unlabeled_source(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    dump_labeled_query_rows(
        Path(str(cfg.teacher_train_jsonl)),
        [_row("q1", "anxiety", "a1")],
    )

    with pytest.raises(ValueError, match="teacher_unlabeled_jsonl"):
        resolve_teacher_and_unlabeled_rows(
            cfg=cfg,
            run_id="bootstrap_v1",
            export_dir=tmp_path / "exports",
            teacher_seed_rows=None,
            teacher_unlabeled_rows=None,
        )
