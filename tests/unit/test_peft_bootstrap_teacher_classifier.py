from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from scripts.support.query_ssl_peft.runners.teacher_classifier import (
    resolve_teacher_classifier,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
)


def _cfg(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "teacher_reuse_manifest_path": "",
            "teacher_embed_chunk_size": 8,
            "teacher_train_batch_size": 4,
            "teacher_eval_batch_size": 4,
            "teacher_epochs": 2,
            "teacher_learning_rate": 0.001,
            "teacher_weight_decay": 0.0001,
            "teacher_output_dir": str(tmp_path / "teacher_runs"),
            "teacher_model_output_dir": str(tmp_path / "teacher_models"),
            "teacher_classifier_version": "",
            "embedding": {
                "cache_dir": "",
                "spec": {"_target_": "builtins.dict"},
            },
            "runtime": {
                "device": "cpu",
                "local_files_only": True,
            },
            "eval_sets": {
                "validation": str(tmp_path / "validation.jsonl"),
            },
            "selection_set": "validation",
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


def test_resolve_teacher_classifier_trains_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    captured: dict[str, object] = {}
    dump_labeled_query_rows(
        Path(str(cfg.eval_sets.validation)),
        [_row("v1", "anxiety", "validation")],
    )

    monkeypatch.setattr(
        "scripts.support.query_ssl_peft.runners.teacher_classifier.instantiate",
        lambda _spec: SimpleNamespace(device="cpu"),
    )

    def _fake_train_fixed_embedding_classifier(**kwargs):
        captured["train_kwargs"] = kwargs
        return SimpleNamespace(
            categories=["anxiety", "depression", "normal", "suicidal"]
        )

    def _fake_write_fixed_classifier_artifacts(**kwargs):
        captured["artifact_kwargs"] = kwargs
        return {
            "output_dir": "runs/fake_teacher",
            "model_path": "data/processed/classifier_heads/fake_teacher.pt",
            "manifest": "data/processed/classifier_heads/fake_teacher.manifest.json",
            "report_json": "runs/fake_teacher/reports/report.json",
        }

    monkeypatch.setattr(
        "scripts.support.query_ssl_peft.runners.teacher_classifier."
        "train_fixed_embedding_classifier",
        _fake_train_fixed_embedding_classifier,
    )
    monkeypatch.setattr(
        "scripts.support.query_ssl_peft.runners.teacher_classifier."
        "write_fixed_classifier_artifacts",
        _fake_write_fixed_classifier_artifacts,
    )

    resolved = resolve_teacher_classifier(
        cfg=cfg,
        run_id="bootstrap_v1",
        generated_at=datetime(2026, 4, 14, 1, 0, tzinfo=timezone.utc),
        seed_rows=[_row("s1", "anxiety", "seed")],
        seed_jsonl_ref="in_memory/teacher_seed_rows.jsonl",
    )

    assert resolved.trained.categories == [
        "anxiety",
        "depression",
        "normal",
        "suicidal",
    ]
    assert resolved.outputs["output_dir"] == "runs/fake_teacher"
    assert captured["train_kwargs"]["selection_set_name"] == "validation"
    assert set(captured["train_kwargs"]["eval_rows_by_name"]) == {"validation"}
    assert captured["artifact_kwargs"]["classifier_version"] == "bootstrap_v1_teacher"
    assert (
        captured["artifact_kwargs"]["train_jsonl_ref"]
        == "in_memory/teacher_seed_rows.jsonl"
    )


def test_resolve_teacher_classifier_reuses_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.teacher_reuse_manifest_path = "data/processed/classifier_heads/canonical.json"
    captured: dict[str, object] = {}

    def _fake_load_fixed_classifier_artifacts(**kwargs):
        captured["load_kwargs"] = kwargs
        return (
            SimpleNamespace(categories=["anxiety", "depression", "normal", "suicidal"]),
            {
                "model_path": "data/processed/classifier_heads/canonical.pt",
                "manifest": cfg.teacher_reuse_manifest_path,
                "report_json": "runs/train_classifier/canonical/reports/report.json",
            },
        )

    monkeypatch.setattr(
        "scripts.support.query_ssl_peft.runners.teacher_classifier."
        "load_fixed_classifier_artifacts",
        _fake_load_fixed_classifier_artifacts,
    )

    resolved = resolve_teacher_classifier(
        cfg=cfg,
        run_id="bootstrap_v1",
        generated_at=datetime(2026, 4, 14, 1, 0, tzinfo=timezone.utc),
        seed_rows=[_row("s1", "anxiety", "seed")],
        seed_jsonl_ref="in_memory/teacher_seed_rows.jsonl",
    )

    assert captured["load_kwargs"] == {
        "manifest_path": cfg.teacher_reuse_manifest_path,
        "device": "cpu",
        "batch_size": 4,
        "cache_dir": None,
        "local_files_only": True,
    }
    assert (
        resolved.outputs["reused_teacher_manifest"] == cfg.teacher_reuse_manifest_path
    )
