"""Query PEFT run artifact path tests."""

from __future__ import annotations

from datetime import datetime, timezone

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from scripts.experiments.query_peft_ssl.io.artifact_paths import (
    build_query_peft_run_artifact_paths,
)


def test_query_ssl_run_output_dir_is_grouped_by_method_name() -> None:
    cfg = OmegaConf.create(
        {
            "ssl_input_mode": "consistency",
            "output_dir": (
                "runs/train_lora_ssl_classifier/consistency/"
                "labeled-ourafla_reddit_unlabeled-ourafla_reddit"
            ),
            "adapter_output_dir": "data/processed/lora_adapters",
            "classifier_output_dir": "data/processed/lora_classifier_heads",
            "query_ssl_method": {
                "name": "fixmatch_usb_v1",
            },
        }
    )

    paths = build_query_peft_run_artifact_paths(
        cfg=cfg,
        trainer_version="lora_fixmatch_2026_05_10_155954",
        created_at=datetime(2026, 5, 10, 15, 59, 54, tzinfo=timezone.utc),
    )

    assert str(paths.output_dir) == (
        "runs/train_lora_ssl_classifier/consistency/"
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit/"
        "fixmatch_usb_v1/lora_fixmatch_2026_05_10_155954"
    )
    assert str(paths.report_path) == (
        "runs/train_lora_ssl_classifier/"
        "consistency/"
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit/"
        "fixmatch_usb_v1/"
        "lora_fixmatch_2026_05_10_155954/"
        "reports/report.json"
    )


def test_pseudo_label_replay_run_output_dir_skips_consistency_method_group() -> None:
    cfg = OmegaConf.create(
        {
            "ssl_input_mode": "pseudo_label_replay",
            "output_dir": (
                "runs/train_lora_ssl_classifier/pseudo_label_replay/"
                "labeled-ourafla_reddit_unlabeled-ourafla_reddit"
            ),
            "adapter_output_dir": "data/processed/lora_adapters",
            "classifier_output_dir": "data/processed/lora_classifier_heads",
            "query_ssl_method": {
                "name": "fixmatch_usb_v1",
            },
        }
    )

    paths = build_query_peft_run_artifact_paths(
        cfg=cfg,
        trainer_version="lora_replay_2026_05_10_155954",
        created_at=datetime(2026, 5, 10, 15, 59, 54, tzinfo=timezone.utc),
    )

    assert str(paths.output_dir) == (
        "runs/train_lora_ssl_classifier/pseudo_label_replay/"
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit/"
        "lora_replay_2026_05_10_155954"
    )


def test_query_ssl_smoke_budget_paths_stay_under_smoke_root() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=["run_controls/central_ssl/budget=smoke"],
        )

    paths = build_query_peft_run_artifact_paths(
        cfg=cfg,
        trainer_version="lora_smoke_2026_05_10_155954",
        created_at=datetime(2026, 5, 10, 15, 59, 54, tzinfo=timezone.utc),
    )

    assert str(paths.output_dir) == (
        "runs/_smoke/train_lora_ssl_classifier/consistency/"
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
        "validation-ourafla_reddit_test-ourafla_reddit/"
        "fixmatch_usb_v1/lora_smoke_2026_05_10_155954"
    )
    assert str(paths.report_path).startswith("runs/_smoke/train_lora_ssl_classifier/")


def test_non_query_ssl_run_output_dir_keeps_flat_run_id() -> None:
    cfg = OmegaConf.create(
        {
            "output_dir": "runs/train_lora_supervised_classifier",
            "adapter_output_dir": "data/processed/lora_adapters",
            "classifier_output_dir": "data/processed/lora_classifier_heads",
        }
    )

    paths = build_query_peft_run_artifact_paths(
        cfg=cfg,
        trainer_version="lora_clf_2026_05_10_155954",
        created_at=datetime(2026, 5, 10, 15, 59, 54, tzinfo=timezone.utc),
    )

    assert str(paths.output_dir) == (
        "runs/train_lora_supervised_classifier/lora_clf_2026_05_10_155954"
    )


def test_supervised_smoke_budget_paths_stay_under_smoke_root() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name=(
                "entrypoints/central_ssl_control/train_lora_supervised_classifier"
            ),
            overrides=["run_controls/central_ssl/budget=smoke"],
        )

    paths = build_query_peft_run_artifact_paths(
        cfg=cfg,
        trainer_version="lora_supervised_smoke_2026_05_10_155954",
        created_at=datetime(2026, 5, 10, 15, 59, 54, tzinfo=timezone.utc),
    )

    assert str(paths.output_dir) == (
        "runs/_smoke/train_lora_supervised_classifier/"
        "lora_supervised_smoke_2026_05_10_155954"
    )
    assert str(paths.report_path).startswith(
        "runs/_smoke/train_lora_supervised_classifier/"
    )
