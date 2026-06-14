from __future__ import annotations

import importlib


def test_run_peft_supervised_control_entrypoint_imports_direct_runner() -> None:
    entrypoint = importlib.import_module(
        "scripts.experiments.central.ssl_control.run_peft_supervised_control"
    )
    runner = importlib.import_module(
        "scripts.support.query_ssl_text_encoder.runners.supervised"
    )

    assert (
        entrypoint.run_supervised_peft_baseline is runner.run_supervised_peft_baseline
    )


def test_run_query_ssl_control_entrypoint_imports_consistency_runner() -> None:
    entrypoint = importlib.import_module(
        "scripts.experiments.central.ssl_control.run_query_ssl_control"
    )
    runner = importlib.import_module(
        "scripts.support.query_ssl_text_encoder.runners.consistency"
    )

    assert entrypoint.run_query_ssl_control is runner.run_query_ssl_control


def test_query_peft_package_keeps_concrete_helpers_out_of_package_surface() -> None:
    package = importlib.import_module("scripts.support.query_ssl_text_encoder")

    assert not hasattr(package, "run_supervised_peft_baseline")
    assert not hasattr(package, "run_query_adaptation_supervised_baseline")
    assert not hasattr(package, "run_fixmatch_peft_baseline")
    assert not hasattr(package, "run_pseudolabel_peft_baseline")
    assert not hasattr(package, "run_pseudo_label_self_training")
