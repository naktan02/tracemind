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
            "query_data_selection": {
                "labeled": "ourafla_reddit",
                "unlabeled": "ourafla_reddit",
            },
            "labeled_exposure_policy": {"name": "client_local_split"},
            "shard_policy": {"name": "dirichlet_label_skew", "alpha": 0.3},
            "query_ssl_method": {"name": "fixmatch_usb_v1"},
            "round_runtime": {
                "adapter_family_name": "peft_classifier",
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
    assert resolve_fl_ssl_split_slug(cfg) == (
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_client_local_seed42"
    )
    assert resolve_fl_ssl_method_composition_slug(cfg) == (
        "fixmatch_usb_v1__peft_classifier__fedavg"
    )
    assert resolve_fl_ssl_run_condition_slug(cfg) == "clients10_rounds50"
    assert str(output_dir) == (
        "runs/fl_ssl/manual_baselines/"
        "fixmatch_usb_v1__peft_classifier__fedavg/"
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_client_local_seed42/"
        "clients10_rounds50/"
        "20260518T010203Z"
    )


def test_fl_ssl_run_condition_slug_records_fedprox_mu() -> None:
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "federated_run_budget": {"client_count": 10, "rounds": 30},
            "query_data_selection": {
                "labeled": "szegeelim_general4",
                "unlabeled": "ourafla_reddit",
            },
            "labeled_exposure_policy": {"name": "shared_client_seed"},
            "query_ssl_method": {"name": "flexmatch_usb_v1"},
            "round_runtime": {
                "adapter_family_name": "peft_classifier",
                "aggregation_backend_name": "fedavg",
            },
            "training_task": {
                "objective": {
                    "peft_classifier": {
                        "proximal_mu": 0.01,
                    }
                }
            },
            "fl_method": {"composition_mode": "manual"},
        }
    )

    output_dir = build_fl_ssl_run_dir(
        "runs/fl_ssl",
        cfg=cfg,
        run_id="20260518T010203Z",
    )

    assert resolve_fl_ssl_run_condition_slug(cfg) == (
        "clients10_rounds30_fedprox_mu0.01"
    )
    assert str(output_dir) == (
        "runs/fl_ssl/manual_baselines/"
        "flexmatch_usb_v1__peft_classifier__fedavg/"
        "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
        "shared_client_seed42/"
        "clients10_rounds30_fedprox_mu0.01/"
        "20260518T010203Z"
    )


def test_fl_ssl_run_condition_slug_keeps_legacy_lora_fedprox_mu() -> None:
    cfg = OmegaConf.create(
        {
            "federated_run_budget": {"client_count": 10, "rounds": 30},
            "training_task": {
                "objective": {
                    "lora_classifier": {
                        "proximal_mu": 0.01,
                    }
                }
            },
        }
    )

    assert resolve_fl_ssl_run_condition_slug(cfg) == (
        "clients10_rounds30_fedprox_mu0.01"
    )


def test_fl_ssl_run_dir_uses_method_owned_family_when_not_manual() -> None:
    cfg = OmegaConf.create(
        {
            "fl_data": {
                "source_mode": "materialized_client_split",
                "split_manifest": (
                    "data/datasets/fl_client_splits/"
                    "client_local_labeled/"
                    "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
                    "validation-ourafla_reddit_test-ourafla_reddit_"
                    "client_local_split_dirichlet_label_skew_dominantNone_"
                    "alpha0.3_clients10_seed42/manifest.json"
                ),
            },
            "seed": 42,
            "federated_run_budget": {"client_count": 10, "rounds": 1},
            "shard_policy": {"name": "dirichlet_label_skew", "alpha": 0.3},
            "ssl_method": {"name": "fedmatch"},
            "server_update_policy": {"name": "fedmatch_partitioned"},
            "round_runtime": {
                "adapter_family_name": "peft_classifier",
                "aggregation_backend_name": "fedavg",
            },
            "fl_method": {"composition_mode": "method_owned"},
        }
    )

    output_dir = build_fl_ssl_run_dir(
        "runs/fl_ssl",
        cfg=cfg,
        run_id="20260518T010203Z",
    )

    assert str(output_dir) == (
        "runs/fl_ssl/fedmatch/"
        "fedmatch__peft_classifier__fedmatch_partitioned/"
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_client_local_seed42/"
        "clients10_rounds1/"
        "20260518T010203Z"
    )


def test_fl_ssl_split_slug_includes_non_default_labeled_exposure_policy() -> None:
    cfg = OmegaConf.create(
        {
            "fl_data": {
                "source_mode": "materialized_client_split",
                "split_manifest": (
                    "data/datasets/fl_client_splits/"
                    "shared_client_labeled/"
                    "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
                    "validation-ourafla_reddit_test-ourafla_reddit_"
                    "labels_pc100_shared_client_seed_dirichlet_label_skew_"
                    "dominantNone_alpha0.3_clients10_seed42/"
                    "manifest.json"
                ),
            },
            "seed": 42,
            "federated_run_budget": {"client_count": 10, "rounds": 1},
            "shard_policy": {"name": "dirichlet_label_skew", "alpha": 0.3},
            "query_ssl_method": {"name": "flexmatch_usb_v1"},
            "round_runtime": {
                "adapter_family_name": "peft_classifier",
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

    assert resolve_fl_ssl_split_slug(cfg) == (
        "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
        "labels_pc100_shared_client_seed42"
    )
    assert str(output_dir) == (
        "runs/fl_ssl/manual_baselines/"
        "flexmatch_usb_v1__peft_classifier__fedavg/"
        "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
        "labels_pc100_shared_client_seed42/"
        "clients10_rounds1/"
        "20260518T010203Z"
    )


def test_fl_ssl_split_slug_compacts_server_only_exposure_name() -> None:
    cfg = OmegaConf.create(
        {
            "fl_data": {
                "source_mode": "materialized_client_split",
                "split_manifest": (
                    "data/datasets/fl_client_splits/"
                    "server_only_labeled/"
                    "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
                    "validation-ourafla_reddit_test-ourafla_reddit_"
                    "server_only_seed_dirichlet_label_skew_dominantNone_"
                    "alpha0.3_clients10_seed42/manifest.json"
                ),
            },
            "seed": 42,
            "labeled_exposure_policy": {"name": "server_only_seed"},
        }
    )

    assert resolve_fl_ssl_split_slug(cfg) == (
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_server_only_seed42"
    )


def test_fl_ssl_split_slug_records_materialized_label_budget() -> None:
    for budget in (25, 100, 400, 1024):
        cfg = OmegaConf.create(
            {
                "fl_data": {
                    "source_mode": "materialized_client_split",
                    "split_manifest": (
                        "data/datasets/fl_client_splits/"
                        "shared_client_labeled/"
                        "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
                        "validation-ourafla_reddit_test-ourafla_reddit_"
                        f"labels_pc{budget}_shared_client_seed_"
                        "dirichlet_label_skew_dominantNone_alpha0.3_"
                        "clients10_seed42/manifest.json"
                    ),
                },
                "seed": 42,
                "labeled_exposure_policy": {"name": "shared_client_seed"},
            }
        )

        assert resolve_fl_ssl_split_slug(cfg) == (
            "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
            f"labels_pc{budget}_shared_client_seed42"
        )


def test_fl_ssl_client_count_sweep_groups_under_method_split_and_rounds() -> None:
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "federated_run_budget": {"rounds": 1},
            "query_data_selection": {
                "labeled": "ourafla_reddit",
                "unlabeled": "ourafla_reddit",
            },
            "labeled_exposure_policy": {"name": "shared_client_seed"},
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
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_shared_client_seed42/"
        "sweeps/client_count_rounds1/"
        "20260518T010203Z"
    )
    assert str(
        build_fl_ssl_client_count_sweep_member_dir(sweep_dir, client_count=10)
    ) == (
        "runs/fl_ssl/manual_baselines/"
        "fixmatch_usb_v1__lora_classifier__fedavg/"
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_shared_client_seed42/"
        "sweeps/client_count_rounds1/"
        "20260518T010203Z/clients_10"
    )
