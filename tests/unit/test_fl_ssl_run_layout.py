"""FL SSL run output layout tests."""

from __future__ import annotations

from omegaconf import OmegaConf

from scripts.experiments.fl_ssl.run_layout import (
    build_fl_ssl_client_count_sweep_member_dir,
    build_fl_ssl_run_dir,
    resolve_fl_ssl_method_composition_slug,
    resolve_fl_ssl_method_family_slug,
    resolve_fl_ssl_run_condition_slug,
    resolve_fl_ssl_split_slug,
)


def test_fl_ssl_run_dir_groups_by_split_and_method_composition() -> None:
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "federated_run_budget": {"client_count": 10, "rounds": 50},
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
        "runs/fl_ssl",
        cfg=cfg,
        run_id="20260518T010203Z",
    )

    assert resolve_fl_ssl_method_family_slug(cfg) == "manual_baselines"
    assert resolve_fl_ssl_split_slug(cfg) == "alpha03_seed42"
    assert resolve_fl_ssl_method_composition_slug(cfg) == (
        "fixmatch_usb_v1__lora_classifier__fedavg"
    )
    assert resolve_fl_ssl_run_condition_slug(cfg) == "clients10_rounds50"
    assert str(output_dir) == (
        "runs/fl_ssl/manual_baselines/"
        "fixmatch_usb_v1__lora_classifier__fedavg/"
        "alpha03_seed42/"
        "clients10_rounds50/"
        "20260518T010203Z"
    )


def test_fl_ssl_run_dir_uses_method_owned_family_when_not_manual() -> None:
    cfg = OmegaConf.create(
        {
            "fl_data": {
                "source_mode": "materialized_client_split",
                "split_manifest": (
                    "data/datasets/fl_client_splits/"
                    "ourafla_dirichlet_alpha03_seed42_clients10/manifest.json"
                ),
            },
            "seed": 42,
            "federated_run_budget": {"client_count": 10, "rounds": 1},
            "shard_policy": {"name": "dirichlet_label_skew", "alpha": 0.3},
            "ssl_method": {"name": "fedmatch"},
            "round_runtime": {
                "adapter_family_name": "lora_classifier",
                "aggregation_backend_name": "fedavg",
            },
        }
    )

    output_dir = build_fl_ssl_run_dir(
        "runs/fl_ssl",
        cfg=cfg,
        run_id="20260518T010203Z",
    )

    assert str(output_dir) == (
        "runs/fl_ssl/fedmatch/"
        "fedmatch__lora_classifier__fedavg/"
        "alpha03_seed42/"
        "clients10_rounds1/"
        "20260518T010203Z"
    )


def test_fl_ssl_client_count_sweep_groups_under_method_split_and_rounds() -> None:
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "federated_run_budget": {"rounds": 1},
            "shard_policy": {"name": "dirichlet_label_skew", "alpha": 0.3},
            "query_ssl_method": {"name": "fixmatch_usb_v1"},
            "round_runtime": {
                "adapter_family_name": "lora_classifier",
                "aggregation_backend_name": "fedavg",
            },
            "fl_method": {"composition_mode": "manual"},
        }
    )

    sweep_dir = build_fl_ssl_run_dir(
        "runs/fl_ssl",
        cfg=cfg,
        run_id="20260518T010203Z",
        run_kind="client_count_sweep",
    )

    assert str(sweep_dir) == (
        "runs/fl_ssl/manual_baselines/"
        "fixmatch_usb_v1__lora_classifier__fedavg/"
        "alpha03_seed42/"
        "sweeps/client_count_rounds1/"
        "20260518T010203Z"
    )
    assert str(
        build_fl_ssl_client_count_sweep_member_dir(sweep_dir, client_count=10)
    ) == (
        "runs/fl_ssl/manual_baselines/"
        "fixmatch_usb_v1__lora_classifier__fedavg/"
        "alpha03_seed42/"
        "sweeps/client_count_rounds1/"
        "20260518T010203Z/clients_10"
    )
