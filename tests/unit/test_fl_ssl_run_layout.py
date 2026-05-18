"""FL SSL run output layout tests."""

from __future__ import annotations

from omegaconf import OmegaConf

from scripts.experiments.fl_ssl.run_layout import (
    build_fl_ssl_run_dir,
    resolve_fl_ssl_method_composition_slug,
    resolve_fl_ssl_split_slug,
)


def test_fl_ssl_run_dir_groups_by_split_and_method_composition() -> None:
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "federated_run_budget": {"client_count": 10},
            "fl_data": {"source_mode": "runtime_split_from_train"},
            "shard_policy": {"name": "dirichlet_label_skew", "alpha": 0.3},
            "query_ssl_method": {"name": "fixmatch_usb_v1"},
            "round_runtime": {
                "adapter_family_name": "lora_classifier",
                "aggregation_backend_name": "fedavg",
            },
            "fl_method": {"composition_mode": "manual"},
        }
    )

    output_dir = build_fl_ssl_run_dir(
        "runs/fl_ssl/single/main",
        cfg=cfg,
        run_id="20260518T010203Z",
    )

    assert resolve_fl_ssl_split_slug(cfg) == (
        "runtime_split_seed42_clients10_dirichlet_label_skew_alpha0p3"
    )
    assert resolve_fl_ssl_method_composition_slug(cfg) == (
        "fixmatch_usb_v1_lora_classifier_fedavg_manual"
    )
    assert str(output_dir) == (
        "runs/fl_ssl/single/main/"
        "runtime_split_seed42_clients10_dirichlet_label_skew_alpha0p3/"
        "fixmatch_usb_v1_lora_classifier_fedavg_manual/"
        "20260518T010203Z"
    )


def test_fl_ssl_run_dir_uses_materialized_split_manifest_parent() -> None:
    cfg = OmegaConf.create(
        {
            "fl_data": {
                "source_mode": "materialized_client_split",
                "split_manifest": (
                    "data/datasets/fl_client_splits/"
                    "ourafla_dirichlet_alpha03_seed42_clients10/manifest.json"
                ),
            },
            "ssl_method": {"name": "fedavg_pseudo_label"},
            "round_runtime": {
                "adapter_family_name": "lora_classifier",
                "aggregation_backend_name": "fedavg",
            },
        }
    )

    output_dir = build_fl_ssl_run_dir(
        "runs/fl_ssl/single/main",
        cfg=cfg,
        run_id="20260518T010203Z",
    )

    assert str(output_dir) == (
        "runs/fl_ssl/single/main/"
        "ourafla_dirichlet_alpha03_seed42_clients10/"
        "fedavg_pseudo_label_lora_classifier_fedavg/"
        "20260518T010203Z"
    )
