"""Query LoRA run artifact path tests."""

from __future__ import annotations

from datetime import datetime, timezone

from omegaconf import OmegaConf

from scripts.experiments.query_lora_ssl.io.artifact_paths import (
    build_query_lora_run_artifact_paths,
)


def test_query_ssl_run_output_dir_is_grouped_by_method_name() -> None:
    cfg = OmegaConf.create(
        {
            "output_dir": "runs/train_lora_query_ssl",
            "adapter_output_dir": "data/processed/lora_adapters",
            "classifier_output_dir": "data/processed/lora_classifier_heads",
            "query_ssl_method": {
                "name": "fixmatch_usb_v1",
            },
        }
    )

    paths = build_query_lora_run_artifact_paths(
        cfg=cfg,
        trainer_version="lora_fixmatch_2026_05_10_155954",
        created_at=datetime(2026, 5, 10, 15, 59, 54, tzinfo=timezone.utc),
    )

    assert str(paths.output_dir) == (
        "runs/train_lora_query_ssl/fixmatch_usb_v1/lora_fixmatch_2026_05_10_155954"
    )
    assert str(paths.report_path) == (
        "runs/train_lora_query_ssl/"
        "fixmatch_usb_v1/"
        "lora_fixmatch_2026_05_10_155954/"
        "reports/report.json"
    )


def test_non_query_ssl_run_output_dir_keeps_flat_run_id() -> None:
    cfg = OmegaConf.create(
        {
            "output_dir": "runs/train_lora_classifier",
            "adapter_output_dir": "data/processed/lora_adapters",
            "classifier_output_dir": "data/processed/lora_classifier_heads",
        }
    )

    paths = build_query_lora_run_artifact_paths(
        cfg=cfg,
        trainer_version="lora_clf_2026_05_10_155954",
        created_at=datetime(2026, 5, 10, 15, 59, 54, tzinfo=timezone.utc),
    )

    assert str(paths.output_dir) == (
        "runs/train_lora_classifier/lora_clf_2026_05_10_155954"
    )
