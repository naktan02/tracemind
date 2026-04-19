from __future__ import annotations

import importlib


def test_train_lora_classifier_entrypoint_imports_direct_runner() -> None:
    entrypoint = importlib.import_module("scripts.experiments.train_lora_classifier")
    runner = importlib.import_module("scripts.experiments.lora_classifier.runner")

    assert (
        entrypoint.run_supervised_lora_baseline
        is runner.run_supervised_lora_baseline
    )


def test_train_lora_pseudo_label_classifier_entrypoint_imports_direct_runner() -> None:
    entrypoint = importlib.import_module(
        "scripts.experiments.train_lora_pseudo_label_classifier"
    )
    runner = importlib.import_module(
        "scripts.experiments.lora_classifier.pseudo_label_runner"
    )

    assert (
        entrypoint.run_pseudo_label_self_training
        is runner.run_pseudo_label_self_training
    )


def test_train_lora_fixmatch_entrypoint_imports_direct_runner() -> None:
    entrypoint = importlib.import_module("scripts.experiments.train_lora_fixmatch")
    runner = importlib.import_module(
        "scripts.experiments.lora_classifier.query_ssl.consistency_runner"
    )

    assert entrypoint.run_fixmatch_lora_baseline is runner.run_fixmatch_lora_baseline


def test_train_lora_bootstrap_classifier_teacher_entrypoint_imports_direct_runner(
) -> None:
    entrypoint = importlib.import_module(
        "scripts.experiments.train_lora_bootstrap_classifier_teacher"
    )
    runner = importlib.import_module(
        "scripts.experiments.lora_classifier.bootstrap_runner"
    )

    assert (
        entrypoint.run_fixed_classifier_teacher_lora_student_bootstrap
        is runner.run_fixed_classifier_teacher_lora_student_bootstrap
    )


def test_lora_classifier_package_keeps_concrete_helpers_out_of_package_surface(
) -> None:
    package = importlib.import_module("scripts.experiments.lora_classifier")

    assert not hasattr(package, "run_supervised_lora_baseline")
    assert not hasattr(package, "run_query_adaptation_supervised_baseline")
    assert not hasattr(package, "run_fixmatch_lora_baseline")
    assert not hasattr(package, "run_pseudo_label_self_training")
