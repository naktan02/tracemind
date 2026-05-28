from __future__ import annotations

import importlib

import pytest
from omegaconf import OmegaConf


def test_train_peft_supervised_classifier_entrypoint_imports_direct_runner() -> None:
    entrypoint = importlib.import_module(
        "scripts.experiments.central_ssl_control.train_peft_supervised_classifier"
    )
    runner = importlib.import_module(
        "scripts.experiments.query_peft_ssl.runners.supervised"
    )

    assert (
        entrypoint.run_supervised_peft_baseline is runner.run_supervised_peft_baseline
    )


def test_train_peft_ssl_classifier_entrypoint_imports_mode_router() -> None:
    entrypoint = importlib.import_module(
        "scripts.experiments.central_ssl_control.train_peft_ssl_classifier"
    )
    router = importlib.import_module(
        "scripts.experiments.central_ssl_control.ssl_mode_router"
    )

    assert entrypoint.run_central_ssl_mode is router.run_central_ssl_mode


def test_central_ssl_mode_router_uses_config_declared_runner(monkeypatch) -> None:
    router = importlib.import_module(
        "scripts.experiments.central_ssl_control.ssl_mode_router"
    )

    captured: dict[str, object] = {}

    def _fake_runner(*, cfg):
        captured["cfg"] = cfg

    monkeypatch.setattr(
        router,
        "load_configured_callable",
        lambda _path, *, field_name: _fake_runner,
    )
    cfg = OmegaConf.create(
        {
            "ssl_input_mode": "consistency",
            "central_ssl_runner": {
                "mode": "consistency",
                "callable_path": "tests.fixtures.central_ssl_runner",
            },
        }
    )

    router.run_central_ssl_mode(cfg)

    assert captured["cfg"] is cfg


def test_central_ssl_mode_router_rejects_scalar_mode_override_without_group() -> None:
    router = importlib.import_module(
        "scripts.experiments.central_ssl_control.ssl_mode_router"
    )
    cfg = OmegaConf.create(
        {
            "ssl_input_mode": "pseudo_label_replay",
            "central_ssl_runner": {
                "mode": "consistency",
                "callable_path": "tests.fixtures.central_ssl_runner",
            },
        }
    )

    with pytest.raises(ValueError, match="strategy_axes/ssl/input_mode"):
        router.run_central_ssl_mode(cfg)


def test_query_peft_package_keeps_concrete_helpers_out_of_package_surface() -> None:
    package = importlib.import_module("scripts.experiments.query_peft_ssl")

    assert not hasattr(package, "run_supervised_peft_baseline")
    assert not hasattr(package, "run_query_adaptation_supervised_baseline")
    assert not hasattr(package, "run_fixmatch_peft_baseline")
    assert not hasattr(package, "run_pseudolabel_peft_baseline")
    assert not hasattr(package, "run_pseudo_label_self_training")
