from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from scripts.experiments.query_peft_ssl.runners.bootstrap_teacher import (
    run_fixed_classifier_teacher_lora_student_bootstrap,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
)

VALIDATION_JSONL = (
    "data/datasets/ourafla_mental_health/splits/ourafla_train_split.v1.validation.jsonl"
)
TEST_JSONL = (
    "data/datasets/ourafla_mental_health/mapped/"
    "ourafla_mental_health_text_classification_test.v1.jsonl"
)


def _build_cfg() -> object:
    return OmegaConf.create(
        {
            "bootstrap_version": "bootstrap_v1",
            "teacher_train_jsonl": "",
            "teacher_unlabeled_jsonl": None,
            "teacher_reuse_manifest_path": "",
            "teacher_embed_chunk_size": 8,
            "teacher_train_batch_size": 4,
            "teacher_eval_batch_size": 4,
            "teacher_epochs": 2,
            "teacher_learning_rate": 0.001,
            "teacher_weight_decay": 0.0001,
            "teacher_output_dir": "runs/train_classifier",
            "teacher_model_output_dir": "data/processed/classifier_heads",
            "teacher_classifier_version": "",
            "bootstrap_split": {
                "enabled": False,
                "unlabeled_ratio": 0.5,
                "seed": 42,
            },
            "embedding": {"spec": {"_target_": "builtins.dict"}},
            "eval_sets": {
                "validation": VALIDATION_JSONL,
                "test": TEST_JSONL,
            },
            "selection_set": "validation",
            "pseudo_label_algorithm": {
                "name": "margin_threshold_test",
                "confidence_threshold": 0.8,
                "margin_threshold": 0.05,
                "algorithm_name": "top1_margin_threshold",
            },
            "bootstrap_export_root": "",
            "pseudo_label_export_root": "",
            "student_include_seed_train_rows": False,
            "fixed_categories": [
                "anxiety",
                "depression",
                "normal",
                "suicidal",
            ],
            "initial_adapter_dir": "",
            "initial_classifier_path": "",
            "runtime": {"device": "cpu"},
            "train_jsonl": "",
            "train_batch_size": 16,
            "eval_batch_size": 32,
            "epochs": 5,
            "learning_rate": 0.0002,
            "classifier_learning_rate": 0.001,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "log_every_steps": 10,
            "ssl_input_mode": "pseudo_label_replay",
            "output_dir": "runs/train_peft_ssl_classifier/pseudo_label_replay",
            "adapter_output_dir": "data/processed/peft_adapters",
            "classifier_output_dir": "data/processed/peft_classifier_heads",
            "trainer_version": "student_v1",
            "seed": 42,
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


def _fake_teacher_classifier() -> SimpleNamespace:
    return SimpleNamespace(
        trained=SimpleNamespace(
            categories=["anxiety", "depression", "normal", "suicidal"]
        ),
        outputs={
            "output_dir": "runs/fake_teacher",
            "model_path": "data/processed/classifier_heads/fake_teacher.pt",
            "manifest": "data/processed/classifier_heads/fake_teacher.manifest.json",
            "report_json": "runs/fake_teacher/reports/report.json",
        },
    )


def test_bootstrap_runner_trains_teacher_then_runs_lora_student(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_cfg()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.runners.bootstrap_teacher."
        "resolve_teacher_classifier",
        lambda **_kwargs: _fake_teacher_classifier(),
    )
    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.runners.bootstrap_teacher."
        "predict_fixed_classifier_rows",
        lambda **_kwargs: [
            SimpleNamespace(
                query_id="u1",
                predicted_label="depression",
                confidence=0.91,
                margin=0.22,
                runner_up_label="suicidal",
                runner_up_score=0.69,
                raw_scores={
                    "anxiety": 0.02,
                    "depression": 0.91,
                    "normal": 0.01,
                    "suicidal": 0.69,
                },
            ),
            SimpleNamespace(
                query_id="u2",
                predicted_label="anxiety",
                confidence=0.62,
                margin=0.01,
                runner_up_label="normal",
                runner_up_score=0.61,
                raw_scores={
                    "anxiety": 0.62,
                    "depression": 0.10,
                    "normal": 0.61,
                    "suicidal": 0.05,
                },
            ),
        ],
    )

    def _fake_student_runner(
        *,
        cfg,
        seed_train_rows,
        pseudo_label_rows,
        include_seed_train_rows,
        train_jsonl_ref,
        trainer_version_override,
        export_root,
        generated_at,
        categories_override,
    ) -> dict[str, str]:
        captured["cfg"] = cfg
        captured["seed_train_rows"] = list(seed_train_rows)
        captured["pseudo_label_rows"] = list(pseudo_label_rows)
        captured["include_seed_train_rows"] = include_seed_train_rows
        captured["train_jsonl_ref"] = train_jsonl_ref
        captured["trainer_version_override"] = trainer_version_override
        captured["export_root"] = export_root
        captured["generated_at"] = generated_at
        captured["categories_override"] = categories_override
        return {
            "output_dir": "runs/fake_student",
            "report_json": "runs/fake_student/reports/report.json",
        }

    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.runners.bootstrap_teacher."
        "run_pseudo_label_self_training",
        _fake_student_runner,
    )

    outputs = run_fixed_classifier_teacher_lora_student_bootstrap(
        cfg=cfg,
        teacher_seed_rows=[_row("s1", "anxiety", "불안해")],
        teacher_unlabeled_rows=[
            _row("u1", "depression", "우울해"),
            _row("u2", "anxiety", "긴장돼"),
        ],
        export_root=tmp_path,
        generated_at=datetime(2026, 4, 14, 1, 0, tzinfo=timezone.utc),
    )

    assert outputs["output_dir"] == "runs/fake_student"
    assert Path(outputs["prediction_trace_jsonl"]).exists()
    assert Path(outputs["prediction_summary_json"]).exists()
    assert Path(outputs["bootstrap_summary"]).exists()
    assert len(captured["seed_train_rows"]) == 1
    assert len(captured["pseudo_label_rows"]) == 1
    assert captured["include_seed_train_rows"] is False
    assert captured["cfg"] is cfg
    assert captured["train_jsonl_ref"] == "in_memory/teacher_seed_rows.jsonl"
    assert captured["trainer_version_override"] == "student_v1"
    assert tuple(captured["categories_override"]) == (
        "anxiety",
        "depression",
        "normal",
        "suicidal",
    )
    assert captured["pseudo_label_rows"][0]["query_id"] == "u1"
    assert captured["pseudo_label_rows"][0]["raw_label_scheme"] == "pseudo_label"

    summary = json.loads(Path(outputs["prediction_summary_json"]).read_text())
    assert summary["accepted_count"] == 1
    assert summary["accepted_hidden_label_accuracy"] == 1.0
    assert summary["pseudo_label_algorithm"]["preset_name"] == "margin_threshold_test"
    assert (
        summary["pseudo_label_algorithm"]["algorithm_name"] == "top1_margin_threshold"
    )


def test_bootstrap_runner_can_auto_split_teacher_seed_and_unlabeled_pool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_cfg()
    cfg.teacher_train_jsonl = str(tmp_path / "full_train.jsonl")
    cfg.bootstrap_split.enabled = True
    cfg.bootstrap_split.unlabeled_ratio = 0.5

    dump_labeled_query_rows(
        Path(cfg.teacher_train_jsonl),
        [
            _row("q1", "anxiety", "a1"),
            _row("q2", "anxiety", "a2"),
            _row("q3", "depression", "d1"),
            _row("q4", "depression", "d2"),
        ],
    )

    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.runners.bootstrap_teacher."
        "resolve_teacher_classifier",
        lambda **_kwargs: _fake_teacher_classifier(),
    )
    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.runners.bootstrap_teacher."
        "predict_fixed_classifier_rows",
        lambda **kwargs: [
            SimpleNamespace(
                query_id=row["query_id"],
                predicted_label=row["mapped_label_4"],
                confidence=0.95,
                margin=0.4,
                runner_up_label="normal",
                runner_up_score=0.1,
                raw_scores={
                    "anxiety": 0.95 if row["mapped_label_4"] == "anxiety" else 0.1,
                    "depression": (
                        0.95 if row["mapped_label_4"] == "depression" else 0.1
                    ),
                    "normal": 0.1,
                    "suicidal": 0.05,
                },
            )
            for row in kwargs["rows"]
        ],
    )
    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.runners.bootstrap_teacher."
        "run_pseudo_label_self_training",
        lambda **_kwargs: {
            "output_dir": "runs/fake_student",
            "report_json": "runs/fake_student/reports/report.json",
        },
    )

    outputs = run_fixed_classifier_teacher_lora_student_bootstrap(
        cfg=cfg,
        export_root=tmp_path / "exports",
        generated_at=datetime(2026, 4, 14, 1, 0, tzinfo=timezone.utc),
    )

    assert Path(outputs["teacher_seed_jsonl"]).exists()
    assert Path(outputs["teacher_unlabeled_jsonl"]).exists()
    assert Path(outputs["teacher_split_manifest"]).exists()


def test_bootstrap_runner_can_reuse_canonical_teacher_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_cfg()
    cfg.teacher_reuse_manifest_path = (
        "data/processed/classifier_heads/canonical.manifest.json"
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.runners.bootstrap_teacher."
        "resolve_teacher_classifier",
        lambda **_kwargs: SimpleNamespace(
            trained=SimpleNamespace(
                categories=["anxiety", "depression", "normal", "suicidal"]
            ),
            outputs={
                "model_path": "data/processed/classifier_heads/canonical.pt",
                "manifest": "data/processed/classifier_heads/canonical.manifest.json",
                "report_json": "runs/train_classifier/canonical/reports/report.json",
                "reused_teacher_manifest": cfg.teacher_reuse_manifest_path,
            },
        ),
    )
    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.runners.bootstrap_teacher."
        "predict_fixed_classifier_rows",
        lambda **_kwargs: [
            SimpleNamespace(
                query_id="u1",
                predicted_label="depression",
                confidence=0.91,
                margin=0.22,
                runner_up_label="suicidal",
                runner_up_score=0.69,
                raw_scores={
                    "anxiety": 0.02,
                    "depression": 0.91,
                    "normal": 0.01,
                    "suicidal": 0.69,
                },
            )
        ],
    )

    def _fake_student_runner(**kwargs) -> dict[str, str]:
        captured["kwargs"] = kwargs
        return {
            "output_dir": "runs/fake_student",
            "report_json": "runs/fake_student/reports/report.json",
        }

    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.runners.bootstrap_teacher."
        "run_pseudo_label_self_training",
        _fake_student_runner,
    )

    outputs = run_fixed_classifier_teacher_lora_student_bootstrap(
        cfg=cfg,
        teacher_seed_rows=[_row("s1", "anxiety", "불안해")],
        teacher_unlabeled_rows=[_row("u1", "depression", "우울해")],
        export_root=tmp_path,
        generated_at=datetime(2026, 4, 14, 1, 0, tzinfo=timezone.utc),
    )

    kwargs = captured["kwargs"]
    assert outputs["reused_teacher_manifest"] == cfg.teacher_reuse_manifest_path
    assert kwargs["cfg"] is cfg
    assert kwargs["include_seed_train_rows"] is False
    assert kwargs["train_jsonl_ref"] == "in_memory/teacher_seed_rows.jsonl"
    assert kwargs["trainer_version_override"] == "student_v1"
    assert tuple(kwargs["categories_override"]) == (
        "anxiety",
        "depression",
        "normal",
        "suicidal",
    )
