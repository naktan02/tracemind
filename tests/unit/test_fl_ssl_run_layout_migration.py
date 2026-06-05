"""FL SSL historical run layout migration 검증."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fl_ssl.migrate_run_layout import (
    build_migration_plan,
    execute_migration_plan,
)


def test_build_migration_plan_maps_legacy_run_to_split_condition_method_layout(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs" / "fl_ssl"
    source_dir = _write_legacy_fl_ssl_run(runs_root, method_name="fixmatch_usb_v1")

    entries = build_migration_plan(runs_root=runs_root)

    assert len(entries) == 1
    assert entries[0].source_dir == source_dir
    assert entries[0].target_dir == (
        runs_root
        / "sz4_ourafla_lp100_shared_s42"
        / "c10_r30_e1_b12_s20"
        / "peft_text_encoder_lora"
        / "fixmatch_fedavg"
        / "20260527T033609Z"
    )


def test_execute_migration_plan_moves_run_dir_and_prunes_empty_parents(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs" / "fl_ssl"
    source_dir = _write_legacy_fl_ssl_run(runs_root, method_name="freematch_usb_v1")
    manifest_path = runs_root / ".layout_migration_test.json"
    entries = build_migration_plan(runs_root=runs_root)

    execute_migration_plan(
        entries=entries,
        manifest_path=manifest_path,
        prune_empty_parents=True,
        runs_root=runs_root,
    )

    target_dir = (
        runs_root
        / "sz4_ourafla_lp100_shared_s42"
        / "c10_r30_e1_b12_s20"
        / "peft_text_encoder_lora"
        / "freematch_fedavg"
        / "20260527T033609Z"
    )
    assert not source_dir.exists()
    assert (target_dir / "reports" / "fl_ssl_main_comparison.report.json").exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["moved_count"] == 1
    assert manifest["entries"][0]["status"] == "moved"
    assert not (runs_root / "manual_baselines").exists()


def _write_legacy_fl_ssl_run(
    runs_root: Path,
    *,
    method_name: str,
) -> Path:
    source_dir = (
        runs_root
        / "manual_baselines"
        / f"{method_name}__peft_classifier_lora__fedavg"
        / (
            "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
            "labels_pc100_shared_client_seed42"
        )
        / "clients10_rounds30"
        / "20260527T033609Z"
    )
    report_path = source_dir / "reports" / "fl_ssl_main_comparison.report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_sample_legacy_report(method_name=method_name), indent=2) + "\n",
        encoding="utf-8",
    )
    return source_dir


def _sample_legacy_report(*, method_name: str) -> dict[str, object]:
    return {
        "schema_version": "federated_simulation_report.v1",
        "track": "fl_ssl_main_comparison",
        "protocol": {
            "seed": 42,
            "client_count": 10,
            "round_budget": 30,
            "fl_method": {
                "composition_mode": "manual",
                "manual_axes": {
                    "client_ssl_objective": method_name.removesuffix("_usb_v1"),
                    "server_aggregation": "fedavg",
                    "update_family": "peft_classifier",
                },
            },
            "round_runtime": {
                "adapter_family_name": "peft_classifier",
                "aggregation_backend_name": "fedavg",
            },
            "local_update_budget": {
                "local_epochs": 1,
                "batch_size": 12,
                "max_steps": 20,
            },
            "fl_data_source": {
                "split_manifest_path": (
                    "data/datasets/fl_client_splits/shared_client_labeled/"
                    "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
                    "validation-ourafla_reddit_test-ourafla_reddit_"
                    "labels_pc100_shared_client_seed_dirichlet_label_skew_"
                    "dominantNone_alpha0.3_clients10_seed42/manifest.json"
                ),
                "source_selection": {
                    "labeled": "szegeelim_general4",
                    "unlabeled": "ourafla_reddit",
                    "validation": "ourafla_reddit",
                    "test": "ourafla_reddit",
                },
                "labeled_policy": {
                    "count_per_class": 100,
                    "mode": "count_per_class",
                },
                "labeled_exposure_policy": {
                    "name": "shared_client_seed",
                },
            },
            "objective": {
                "query_ssl.method_name": method_name,
                "query_ssl.algorithm_name": method_name.removesuffix("_usb_v1"),
                "peft_classifier.peft_adapter_name": "lora",
                "peft_classifier.proximal_mu": 0.0,
            },
        },
    }
