from __future__ import annotations

import importlib


def test_train_lora_classifier_entrypoint_imports_direct_runner() -> None:
    entrypoint = importlib.import_module("scripts.experiments.train_lora_classifier")
    runner = importlib.import_module("scripts.experiments.lora_classifier.runner")

    assert entrypoint.run_supervised_lora_baseline is runner.run_supervised_lora_baseline


def test_lora_classifier_package_keeps_concrete_helpers_out_of_package_surface() -> None:
    package = importlib.import_module("scripts.experiments.lora_classifier")

    assert not hasattr(package, "run_supervised_lora_baseline")
    assert not hasattr(package, "run_query_adaptation_supervised_baseline")
