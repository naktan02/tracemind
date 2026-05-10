from __future__ import annotations

import torch
from omegaconf import OmegaConf

from scripts.experiments.query_lora_ssl.runners.consistency import (
    run_fixmatch_lora_baseline,
    run_pseudolabel_lora_baseline,
    run_query_ssl_lora_baseline,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

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
                "supervised_loss_weight": 1.0,
                "unlabeled_batch_size": 4,
                "require_multiview": True,
            },
            "query_ssl_augmenter": {
                "name": "backtranslation_nllb_en_de_fr_usb_v1",
                "augmenter_type": "nllb_backtranslation",
                "source_lang": "eng_Latn",
                "pivot_languages": ["deu_Latn", "fra_Latn"],
                "model_id": "facebook/nllb-200-distilled-600M",
                "revision": "main",
                "device": "cpu",
                "local_files_only": True,
                "batch_size": 8,
                "max_new_tokens": 256,
                "cache_dir": "",
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
            "output_dir": "runs/train_lora_query_ssl",
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


def _usb_unlabeled_row(query_id: str, label: str, text: str) -> LabeledQueryRow:
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
        aug_0=f"de::{text}",
        aug_1=f"fr::{text}",
        aug_0_pivot_lang="deu_Latn",
        aug_1_pivot_lang="fra_Latn",
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

    def _fake_train_query_ssl_classifier(**kwargs):
        captured["algorithm"] = kwargs["algorithm"]
        captured["train_loader"] = kwargs["train_loader"]
        captured["unlabeled_loader"] = kwargs["unlabeled_loader"]
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
        return {
            "output_dir": "runs/fake_fixmatch",
            "report_json": "runs/fake_fixmatch/report.json",
        }

    monkeypatch.setattr(
        "scripts.experiments.query_lora_ssl.harness.common.build_query_lora_model",
        lambda **_kwargs: (
            _DummyModel(),
            _DummyTokenizer(),
            {"parameter_counts": {"trainable": 10, "total": 20}},
        ),
    )
    monkeypatch.setattr(
        "scripts.experiments.query_lora_ssl.runners.consistency."
        "train_query_ssl_lora_classifier",
        _fake_train_query_ssl_classifier,
    )
    monkeypatch.setattr(
        "scripts.experiments.query_lora_ssl.harness.common.evaluate_query_lora_classifier",
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
        "scripts.experiments.query_lora_ssl.runners.consistency.write_run_artifacts",
        _fake_write_run_artifacts,
    )

    outputs = run_fixmatch_lora_baseline(
        cfg=_build_cfg(),
        train_rows=[_labeled_row("seed_q1", "anxiety", "불안해요")],
        unlabeled_rows=[_usb_unlabeled_row("u1", "depression", "우울해요")],
        eval_rows_by_name={
            "validation": [_labeled_row("v1", "anxiety", "검증")],
            "test": [_labeled_row("t1", "depression", "테스트")],
        },
    )

    assert outputs["output_dir"] == "runs/fake_fixmatch"
    assert captured["history"] == [{"epoch": 1, "train_loss": 0.1}]
    assert captured["algorithm"].p_cutoff == 0.95
    assert captured["algorithm"].hard_label is True
    assert captured["algorithm"].supervised_loss_weight == 1.0
    assert captured["extra_manifest"]["unlabeled_row_count"] == 1
    assert (
        captured["extra_manifest"]["query_ssl_method"]["preset_name"]
        == "fixmatch_usb_v1"
    )
    assert (
        captured["extra_manifest"]["query_ssl_method"]["algorithm_name"] == "fixmatch"
    )
    assert (
        captured["extra_manifest"]["query_ssl_method"]["supervised_loss_weight"] == 1.0
    )
    assert (
        captured["extra_manifest"]["query_ssl_method"]["parameters"]["p_cutoff"] == 0.95
    )
    assert (
        captured["extra_manifest"]["query_ssl_augmenter"]["preset_name"]
        == "backtranslation_nllb_en_de_fr_usb_v1"
    )
    assert (
        captured["extra_manifest"]["query_ssl_augmenter_preparation"]["mode"]
        == "precomputed_usb_candidates"
    )
    runtime_metrics = captured["extra_manifest"]["runtime_metrics"]
    assert runtime_metrics["training_example_count"] == 2
    assert runtime_metrics["train_seconds"] > 0
    assert runtime_metrics["examples_per_second"] > 0
    assert runtime_metrics["trainable_param_ratio"] == 0.5


def test_run_query_ssl_lora_baseline_uses_methods_descriptor(
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

    def _fake_train_query_ssl_classifier(**kwargs):
        captured["algorithm"] = kwargs["algorithm"]
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

    monkeypatch.setattr(
        "scripts.experiments.query_lora_ssl.harness.common.build_query_lora_model",
        lambda **_kwargs: (
            _DummyModel(),
            _DummyTokenizer(),
            {"parameter_counts": {"trainable": 10, "total": 20}},
        ),
    )
    monkeypatch.setattr(
        "scripts.experiments.query_lora_ssl.runners.consistency."
        "train_query_ssl_lora_classifier",
        _fake_train_query_ssl_classifier,
    )
    monkeypatch.setattr(
        "scripts.experiments.query_lora_ssl.harness.common.evaluate_query_lora_classifier",
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
        "scripts.experiments.query_lora_ssl.runners.consistency.write_run_artifacts",
        lambda **_kwargs: {
            "output_dir": "runs/fake_fixmatch",
            "report_json": "runs/fake_fixmatch/report.json",
        },
    )

    run_query_ssl_lora_baseline(
        cfg=_build_cfg(),
        train_rows=[_labeled_row("seed_q1", "anxiety", "불안해요")],
        unlabeled_rows=[_usb_unlabeled_row("u1", "depression", "우울해요")],
        eval_rows_by_name={
            "validation": [_labeled_row("v1", "anxiety", "검증")],
            "test": [_labeled_row("t1", "depression", "테스트")],
        },
    )

    assert captured["algorithm"].algorithm_name == "fixmatch"
    assert captured["algorithm"].uses_labeled_batches is True


def test_run_pseudolabel_lora_baseline_uses_weak_text_without_augmentation(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    cfg = _build_cfg()
    cfg.query_ssl_method = OmegaConf.create(
        {
            "name": "pseudolabel_usb_v1",
            "algorithm_name": "pseudolabel",
            "p_cutoff": 0.95,
            "unsup_warm_up": 0.4,
            "lambda_u": 1.0,
            "supervised_loss_weight": 1.0,
            "unlabeled_batch_size": 4,
            "require_multiview": False,
        }
    )

    class _DummyModel:
        pass

    class _DummyTokenizer:
        def __call__(self, texts, **_kwargs):
            batch = len(texts)
            return {
                "input_ids": torch.ones((batch, 2), dtype=torch.long),
                "attention_mask": torch.ones((batch, 2), dtype=torch.long),
            }

    def _fake_train_query_ssl_classifier(**kwargs):
        captured["algorithm"] = kwargs["algorithm"]
        captured["unlabeled_loader"] = kwargs["unlabeled_loader"]
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
        return {
            "output_dir": "runs/fake_pseudolabel",
            "report_json": "runs/fake_pseudolabel/report.json",
        }

    monkeypatch.setattr(
        "scripts.experiments.query_lora_ssl.harness.common.build_query_lora_model",
        lambda **_kwargs: (
            _DummyModel(),
            _DummyTokenizer(),
            {"parameter_counts": {"trainable": 10, "total": 20}},
        ),
    )
    monkeypatch.setattr(
        "scripts.experiments.query_lora_ssl.runners.consistency."
        "train_query_ssl_lora_classifier",
        _fake_train_query_ssl_classifier,
    )
    monkeypatch.setattr(
        "scripts.experiments.query_lora_ssl.harness.common.evaluate_query_lora_classifier",
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
        "scripts.experiments.query_lora_ssl.runners.consistency.write_run_artifacts",
        _fake_write_run_artifacts,
    )

    outputs = run_pseudolabel_lora_baseline(
        cfg=cfg,
        train_rows=[_labeled_row("seed_q1", "anxiety", "불안해요")],
        unlabeled_rows=[_labeled_row("u1", "depression", "우울해요")],
        eval_rows_by_name={
            "validation": [_labeled_row("v1", "anxiety", "검증")],
            "test": [_labeled_row("t1", "depression", "테스트")],
        },
    )
    unlabeled_batch = next(iter(captured["unlabeled_loader"]))

    assert outputs["output_dir"] == "runs/fake_pseudolabel"
    assert captured["algorithm"].algorithm_name == "pseudolabel"
    assert captured["algorithm"].unsup_warm_up == 0.4
    assert "strong_input_ids" not in unlabeled_batch
    assert (
        captured["extra_manifest"]["query_ssl_method"]["preset_name"]
        == "pseudolabel_usb_v1"
    )
    assert (
        captured["extra_manifest"]["query_ssl_augmenter_preparation"]["mode"]
        == "raw_weak_text"
    )
    assert "query_ssl_augmenter" not in captured["extra_manifest"]


def test_run_fixmatch_lora_baseline_rejects_unlabeled_rows_without_usb_candidates_when_precomputed_only() -> (  # noqa: E501
    None
):
    cfg = _build_cfg()
    cfg.query_ssl_augmenter = OmegaConf.create(
        {
            "name": "precomputed_usb_candidates_v1",
            "augmenter_type": "precomputed_usb_candidates",
            "cache_dir": "",
        }
    )

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
        assert "aug_0 and aug_1" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError(
            "FixMatch runner should require strict USB aug candidates when "
            "precomputed-only augmentation is selected."
        )
