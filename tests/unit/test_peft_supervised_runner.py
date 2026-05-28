from __future__ import annotations

import torch
from omegaconf import OmegaConf

from scripts.experiments.query_peft_ssl.runners.supervised import (
    run_supervised_peft_baseline,
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
            "trainer_version": "supervised_run_v1",
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
            "learning_rate": 0.0002,
            "classifier_learning_rate": 0.001,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "log_every_steps": 10,
            "output_dir": "runs/run_peft_supervised_control",
            "adapter_output_dir": "data/processed/peft_adapters",
            "classifier_output_dir": "data/processed/peft_classifier_heads",
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


def test_run_supervised_peft_baseline_wires_common_context_and_artifacts(
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

    def _fake_train_classifier(**kwargs):
        captured["train_loader"] = kwargs["train_loader"]
        captured["selection_loader"] = kwargs["selection_loader"]
        captured["categories"] = kwargs["categories"]
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

    def _fake_write_run_artifacts(**kwargs):
        captured["extra_manifest"] = kwargs["extra_manifest"]
        captured["history"] = kwargs["history"]
        captured["trainer_version"] = kwargs["trainer_version"]
        return {
            "output_dir": "runs/fake_supervised",
            "report_json": "runs/fake_supervised/report.json",
        }

    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.harness.common.build_query_peft_model",
        lambda **_kwargs: (
            _DummyModel(),
            _DummyTokenizer(),
            {"parameter_counts": {"trainable": 10, "total": 20}},
        ),
    )
    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.runners.supervised.train_query_peft_classifier",
        _fake_train_classifier,
    )
    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.harness.common.evaluate_query_peft_classifier",
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
        "scripts.experiments.query_peft_ssl.runners.supervised.write_run_artifacts",
        _fake_write_run_artifacts,
    )

    outputs = run_supervised_peft_baseline(
        cfg=_build_cfg(),
        train_rows=[_labeled_row("seed_q1", "anxiety", "불안해요")],
        eval_rows_by_name={
            "validation": [_labeled_row("v1", "anxiety", "검증")],
            "test": [_labeled_row("t1", "depression", "테스트")],
        },
        extra_manifest={"experiment_family": "supervised"},
    )

    assert outputs["output_dir"] == "runs/fake_supervised"
    assert captured["history"] == [{"epoch": 1, "train_loss": 0.1}]
    assert captured["trainer_version"] == "supervised_run_v1"
    assert captured["categories"] == [
        "anxiety",
        "depression",
        "normal",
        "suicidal",
    ]
    assert captured["extra_manifest"]["experiment_family"] == "supervised"
    assert (
        captured["extra_manifest"]["query_adaptation_initial_checkpoint"]["mode"]
        == "none"
    )
    runtime_metrics = captured["extra_manifest"]["runtime_metrics"]
    assert runtime_metrics["training_example_count"] == 1
    assert runtime_metrics["train_seconds"] > 0
    assert runtime_metrics["examples_per_second"] > 0
    assert runtime_metrics["trainable_param_ratio"] == 0.5
