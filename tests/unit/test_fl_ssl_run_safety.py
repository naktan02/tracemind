"""FL SSL long-run safety guard tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from scripts.experiments.fl_ssl.federated_simulation.sweep import (
    run_client_count_sweep_from_config,
    run_seed_sweep_from_config,
)
from scripts.experiments.fl_ssl.run_federated_simulation import (
    resolve_single_simulation_output_dir,
)
from scripts.experiments.fl_ssl.support.safety import (
    DEFAULT_LONG_RUN_ACK,
    require_fl_ssl_run_budget_allowed,
)


def test_fl_ssl_long_run_guard_allows_default_smoke_budget() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/fl_ssl/run_federated_simulation")

    require_fl_ssl_run_budget_allowed(cfg, run_kind="single_simulation")


def test_fl_ssl_long_run_guard_allows_main_budget_without_ack() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=["run_controls/fl_ssl/budget=main"],
        )

    require_fl_ssl_run_budget_allowed(cfg, run_kind="single_simulation")


def test_fl_ssl_long_run_guard_blocks_rounds_above_main_budget_without_ack() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "run_controls/fl_ssl/budget=main",
                "federated_run_budget.rounds=31",
            ],
        )

    with pytest.raises(ValueError, match="total_planned_rounds=31"):
        require_fl_ssl_run_budget_allowed(cfg, run_kind="single_simulation")


def test_fl_ssl_long_run_guard_allows_reduced_single_run_budget() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=["run_controls/fl_ssl/budget=reduced"],
        )

    require_fl_ssl_run_budget_allowed(cfg, run_kind="single_simulation")


def test_fl_ssl_single_run_output_dir_uses_layout_slug_and_run_id() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/fl_ssl/run_federated_simulation")

    output_dir = resolve_single_simulation_output_dir(
        cfg,
        created_at=datetime(2026, 6, 5, 12, 34, 56, tzinfo=timezone.utc),
    )

    assert output_dir.as_posix() == (
        "runs/_smoke/fl_ssl/"
        "sz4_ourafla_shared_s42/"
        "c4_r3_e1_b8_s50/"
        "peft_text_encoder_lora/"
        "fixmatch_fedavg/"
        "20260605T123456Z"
    )


def test_fl_ssl_single_run_output_dir_uses_resume_dir_when_enabled() -> None:
    cfg = OmegaConf.create(
        {
            "resume": {
                "enabled": True,
                "run_dir": "runs/fl_ssl/existing_run",
            }
        }
    )

    assert (
        resolve_single_simulation_output_dir(cfg).as_posix()
        == "runs/fl_ssl/existing_run"
    )


def test_fl_ssl_long_run_guard_requires_exact_ack() -> None:
    cfg = _minimal_cfg(
        rounds=50,
        run_safety={
            "allow_long_run": True,
            "long_run_ack": "wrong",
            "required_long_run_ack": DEFAULT_LONG_RUN_ACK,
        },
    )

    with pytest.raises(ValueError, match="ALLOW_FL_SSL_LONG_RUN"):
        require_fl_ssl_run_budget_allowed(cfg, run_kind="single_simulation")

    cfg.run_safety.long_run_ack = DEFAULT_LONG_RUN_ACK
    require_fl_ssl_run_budget_allowed(cfg, run_kind="single_simulation")


def test_client_count_sweep_guard_blocks_total_rounds_before_running() -> None:
    cfg = _minimal_cfg(
        rounds=5,
        client_counts=list(range(1, 11)),
    )

    with pytest.raises(ValueError, match="client_count_sweep"):
        run_client_count_sweep_from_config(
            cfg,
            line_renderer=lambda **_: [],
        )


def test_seed_sweep_guard_blocks_total_rounds_before_running() -> None:
    cfg = _minimal_cfg(
        rounds=17,
        seeds=[42, 43, 44],
        seed_count=3,
    )

    with pytest.raises(ValueError, match="seed_sweep"):
        run_seed_sweep_from_config(
            cfg,
            line_renderer=lambda **_: [],
        )


def _minimal_cfg(
    *,
    rounds: int,
    client_counts: list[int] | None = None,
    seeds: list[int] | None = None,
    seed_count: int = 3,
    run_safety: dict[str, object] | None = None,
):
    return OmegaConf.create(
        {
            "seed": 42,
            "report": {
                "track": "fl_ssl_main_comparison",
                "seed_count": seed_count,
            },
            "federated_run_budget": {
                "client_count": 10,
                "rounds": rounds,
            },
            "fl_data": {
                "source_mode": "runtime_split_from_train",
                "split_manifest": None,
            },
            "query_ssl_method": {"name": "fixmatch_usb_v1"},
            "fl_method": {"composition_mode": "manual"},
            "shard_policy": {"name": "dirichlet_label_skew"},
            "client_pool_split": {
                "labeled_ratio": 0.1,
                "unlabeled_ratio": 0.9,
            },
            "sweep": {
                "axis": "none",
                "output_dir": "runs/test_sweep",
                "client_count": {
                    "members": client_counts or [1],
                    "split_manifest_by_client_count": None,
                },
                "seed": {
                    "members": seeds or [42, 43, 44],
                },
            },
            "run_safety": run_safety or {},
        }
    )
