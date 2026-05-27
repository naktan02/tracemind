from __future__ import annotations

import importlib


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


def test_central_ssl_mode_router_imports_consistency_runner() -> None:
    router = importlib.import_module(
        "scripts.experiments.central_ssl_control.ssl_mode_router"
    )
    runner = importlib.import_module(
        "scripts.experiments.query_peft_ssl.runners.consistency"
    )

    assert router.run_query_ssl_peft_baseline is runner.run_query_ssl_peft_baseline


def test_central_ssl_mode_router_imports_pseudo_label_runner() -> None:
    router = importlib.import_module(
        "scripts.experiments.central_ssl_control.ssl_mode_router"
    )
    runner = importlib.import_module(
        "scripts.experiments.query_peft_ssl.runners.pseudo_label"
    )
    assert (
        router.run_pseudo_label_self_training is runner.run_pseudo_label_self_training
    )


def test_query_peft_package_keeps_concrete_helpers_out_of_package_surface() -> None:
    package = importlib.import_module("scripts.experiments.query_peft_ssl")

    assert not hasattr(package, "run_supervised_peft_baseline")
    assert not hasattr(package, "run_query_adaptation_supervised_baseline")
    assert not hasattr(package, "run_fixmatch_peft_baseline")
    assert not hasattr(package, "run_pseudolabel_peft_baseline")
    assert not hasattr(package, "run_pseudo_label_self_training")
