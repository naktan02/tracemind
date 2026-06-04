from __future__ import annotations

import torch
from omegaconf import OmegaConf

from scripts.support.query_ssl_text_encoder.runners import (
    full_text_encoder_supervised as full_text_encoder_supervised_runner,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

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
            "trainer_version": "full_supervised_run_v1",
            "runtime": {"device": "cpu"},
            "train_jsonl": "",
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
            "learning_rate": 0.00002,
            "classifier_learning_rate": 0.0002,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "log_every_steps": 10,
            "output_dir": "runs/run_full_text_encoder_supervised_control",
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
        created_at="2026-04-22T00:00:00+00:00",
    )


def test_run_full_text_encoder_supervised_baseline_uses_common_context(
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

    def _fake_build_model(**_kwargs):
        return (
            _DummyModel(),
            _DummyTokenizer(),
            {
                "parameter_counts": {"trainable": 30, "total": 30},
                "trainable_surface": {"name": "full_text_encoder"},
            },
        )

    def _fake_train_classifier(**kwargs):
        captured["categories"] = kwargs["categories"]
        captured["learning_rate"] = kwargs["learning_rate"]
        captured["classifier_learning_rate"] = kwargs["classifier_learning_rate"]
        return (
            kwargs["model"],
            [{"epoch": 1, "train_loss": 0.1}],
            {
                "loss": 0.2,
                "accuracy_top_1": 0.75,
                "rows_total": 2,
                "mean_true_label_probability": 0.7,
                "mean_top_1_probability": 0.8,
                "mean_margin_top1_top2": 0.3,
                "confusion_matrix": {},
                "per_category": {},
            },
        )

    def _fake_write_artifacts(**kwargs):
        captured["extra_manifest"] = kwargs["extra_manifest"]
        captured["history"] = kwargs["history"]
        captured["trainer_version"] = kwargs["trainer_version"]
        captured["backbone_summary"] = kwargs["backbone_summary"]
        return {
            "output_dir": "runs/fake_full_supervised",
            "report_json": "runs/fake_full_supervised/report.json",
        }

    monkeypatch.setattr(
        "scripts.support.query_ssl_text_encoder.runners.full_text_encoder_supervised."
        "build_full_text_encoder_model",
        _fake_build_model,
    )
    monkeypatch.setattr(
        "scripts.support.query_ssl_text_encoder.runners.full_text_encoder_supervised."
        "train_text_encoder_classifier",
        _fake_train_classifier,
    )
    monkeypatch.setattr(
        "scripts.support.query_ssl_text_encoder.text_encoder_run_context.evaluate_text_encoder_classifier",
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
        "scripts.support.query_ssl_text_encoder.runners.full_text_encoder_supervised."
        "write_full_text_encoder_run_artifacts",
        _fake_write_artifacts,
    )

    outputs = (
        full_text_encoder_supervised_runner.run_full_text_encoder_supervised_baseline(
            cfg=_build_cfg(),
            train_rows=[_labeled_row("seed_q1", "anxiety", "불안해요")],
            eval_rows_by_name={
                "validation": [_labeled_row("v1", "anxiety", "검증")],
                "test": [_labeled_row("t1", "depression", "테스트")],
            },
            extra_manifest={"experiment_family": "full_supervised"},
        )
    )

    assert outputs["output_dir"] == "runs/fake_full_supervised"
    assert captured["history"] == [{"epoch": 1, "train_loss": 0.1}]
    assert captured["trainer_version"] == "full_supervised_run_v1"
    assert captured["categories"] == [
        "anxiety",
        "depression",
        "normal",
        "suicidal",
    ]
    assert captured["learning_rate"] == 0.00002
    assert captured["classifier_learning_rate"] == 0.0002
    assert captured["backbone_summary"]["trainable_surface"] == {
        "name": "full_text_encoder"
    }
    assert captured["extra_manifest"]["experiment_family"] == "full_supervised"
    runtime_metrics = captured["extra_manifest"]["runtime_metrics"]
    assert runtime_metrics["training_example_count"] == 1
    assert runtime_metrics["trainable_param_ratio"] == 1.0
