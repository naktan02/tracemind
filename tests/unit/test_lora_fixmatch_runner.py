from __future__ import annotations

import torch
from omegaconf import OmegaConf

from scripts.experiments.lora_classifier.fixmatch_runner import (
    run_fixmatch_lora_baseline,
)
from scripts.labeled_query_rows import LabeledQueryRow

VALIDATION_JSONL = "data/processed/splits/ourafla_train_split.v1.validation.jsonl"
TEST_JSONL = (
    "data/processed/labeled_query_sets/"
    "ourafla_mental_health_text_classification_test.v1.jsonl"
)


def _build_cfg() -> object:
    return OmegaConf.create(
        {
            "trainer_version": "fixmatch_run_v1",
            "runtime": {"device": "cpu"},
            "train_jsonl": "",
            "unlabeled_jsonl": "",
            "query_ssl_method": {
                "name": "fixmatch_usb_v1",
                "algorithm_name": "fixmatch",
                "temperature": 0.5,
                "p_cutoff": 0.95,
                "hard_label": True,
                "lambda_u": 1.0,
                "unlabeled_batch_size": 4,
                "require_multiview": True,
            },
            "fixed_categories": [
                "anxiety",
                "depression",
                "normal",
                "suicidal",
            ],
            "paper_backbone": {
                "max_length": 32,
                "task_prefix": "",
            },
            "eval_sets": {
                "validation": VALIDATION_JSONL,
                "test": TEST_JSONL,
            },
            "selection_set": "validation",
            "seed": 42,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.0002,
            "classifier_learning_rate": 0.001,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "log_every_steps": 10,
            "output_dir": "runs/train_lora_fixmatch",
            "adapter_output_dir": "data/processed/lora_adapters",
            "classifier_output_dir": "data/processed/lora_classifier_heads",
        }
    )


def _labeled_row(query_id: str, label: str, text: str) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=text,
        raw_label_scheme="manual_label",
        raw_label=label,
        mapped_label_4=label,
        locale="ko-KR",
        annotation_source="seed_train",
        approved_by="annotator",
        created_at="2026-04-19T00:00:00+00:00",
    )


def _multiview_unlabeled_row(query_id: str, label: str, text: str) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=text,
        raw_label_scheme="query_buffer",
        raw_label=label,
        mapped_label_4=label,
        locale="ko-KR",
        annotation_source="query_unlabeled",
        approved_by=None,
        created_at="2026-04-19T00:00:00+00:00",
        weak_text=f"weak::{text}",
        strong_text=f"strong::{text}",
    )


def test_run_fixmatch_lora_baseline_wires_usb_method_manifest(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyModel:
        pass

    class _DummyTokenizer:
        def __call__(self, texts, **_kwargs):
            batch = len(texts)
            return {
                "input_ids": torch.ones((batch, 2), dtype=torch.long),
                "attention_mask": torch.ones((batch, 2), dtype=torch.long),
            }

    def _fake_train_fixmatch_classifier(**kwargs):
        captured["fixmatch_config"] = kwargs["fixmatch_config"]
        captured["train_loader"] = kwargs["train_loader"]
        captured["unlabeled_loader"] = kwargs["unlabeled_loader"]
        return kwargs["model"], [{"epoch": 1, "train_loss": 0.1}], {
            "loss": 0.2,
            "accuracy_top_1": 0.75,
            "rows_total": 2,
            "mean_true_label_probability": 0.7,
            "mean_top_1_probability": 0.8,
            "mean_margin_top1_top2": 0.3,
            "confusion_matrix": {},
            "per_category": {},
        }

    def _fake_write_run_artifacts(**kwargs):
        captured["extra_manifest"] = kwargs["extra_manifest"]
        captured["history"] = kwargs["history"]
        return {
            "output_dir": "runs/fake_fixmatch",
            "report_json": "runs/fake_fixmatch/report.json",
        }

    monkeypatch.setattr(
        "scripts.experiments.lora_classifier.fixmatch_runner.build_model",
        lambda **_kwargs: (
            _DummyModel(),
            _DummyTokenizer(),
            {"parameter_counts": {"trainable": 10, "total": 20}},
        ),
    )
    monkeypatch.setattr(
        "scripts.experiments.lora_classifier.fixmatch_runner.train_fixmatch_classifier",
        _fake_train_fixmatch_classifier,
    )
    monkeypatch.setattr(
        "scripts.experiments.lora_classifier.fixmatch_runner.evaluate_classifier",
        lambda **_kwargs: {
            "loss": 0.1,
            "accuracy_top_1": 0.8,
            "rows_total": 2,
            "mean_true_label_probability": 0.75,
            "mean_top_1_probability": 0.85,
            "mean_margin_top1_top2": 0.4,
            "confusion_matrix": {},
            "per_category": {},
        },
    )
    monkeypatch.setattr(
        "scripts.experiments.lora_classifier.fixmatch_runner.write_run_artifacts",
        _fake_write_run_artifacts,
    )

    outputs = run_fixmatch_lora_baseline(
        cfg=_build_cfg(),
        train_rows=[_labeled_row("seed_q1", "anxiety", "불안해요")],
        unlabeled_rows=[_multiview_unlabeled_row("u1", "depression", "우울해요")],
        eval_rows_by_name={
            "validation": [_labeled_row("v1", "anxiety", "검증")],
            "test": [_labeled_row("t1", "depression", "테스트")],
        },
    )

    assert outputs["output_dir"] == "runs/fake_fixmatch"
    assert captured["history"] == [{"epoch": 1, "train_loss": 0.1}]
    assert captured["fixmatch_config"].p_cutoff == 0.95
    assert captured["fixmatch_config"].hard_label is True
    assert captured["extra_manifest"]["unlabeled_row_count"] == 1
    assert (
        captured["extra_manifest"]["query_ssl_method"]["preset_name"]
        == "fixmatch_usb_v1"
    )
    assert (
        captured["extra_manifest"]["query_ssl_method"]["algorithm_name"] == "fixmatch"
    )


def test_run_fixmatch_lora_baseline_rejects_unlabeled_rows_without_multiview() -> None:
    cfg = _build_cfg()

    unlabeled_rows = [_labeled_row("u1", "depression", "우울해요")]

    try:
        run_fixmatch_lora_baseline(
            cfg=cfg,
            train_rows=[_labeled_row("seed_q1", "anxiety", "불안해요")],
            unlabeled_rows=unlabeled_rows,
            eval_rows_by_name={
                "validation": [_labeled_row("v1", "anxiety", "검증")],
                "test": [_labeled_row("t1", "depression", "테스트")],
            },
        )
    except ValueError as exc:
        assert "weak_text and strong_text" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("FixMatch runner should require multiview unlabeled rows.")
