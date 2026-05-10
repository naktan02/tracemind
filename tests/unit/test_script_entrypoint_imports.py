"""주요 script entrypoint import smoke tests."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.experiments.prototype_analysis.prototype_strategy_experiment",
        "scripts.experiments.prototype_analysis.prototype_threshold_sweep",
        "scripts.experiments.fl_ssl.run_federated_simulation",
        "scripts.experiments.fl_ssl.run_federated_seed_sweep",
        "scripts.experiments.central_classifier_seed.train_softmax_classifier",
        "scripts.experiments.central_ssl_control.train_lora_pseudolabel",
        "scripts.experiments.central_ssl_control.train_lora_fixmatch",
        "scripts.experiments.central_ssl_control.train_lora_pseudo_label_classifier",
        "scripts.experiments.central_ssl_control.train_lora_bootstrap_classifier_teacher",
    ],
)
def test_experiment_entrypoints_import_without_symbol_errors(
    module_name: str,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    module = importlib.import_module(module_name)

    assert module is not None
