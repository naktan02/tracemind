"""주요 script entrypoint import smoke tests."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.experiments.prototype_analysis.prototype_strategy_experiment",
        "scripts.experiments.prototype_analysis.prototype_threshold_sweep",
        "scripts.experiments.fl_ssl.materialize_fl_client_split",
        "scripts.experiments.fl_ssl.run_federated_simulation",
        "scripts.experiments.fl_ssl.run_federated_seed_sweep",
        "scripts.experiments.fl_ssl.run_federated_client_count_sweep",
        "scripts.experiments.central_classifier_seed.train_softmax_classifier",
        "scripts.experiments.central_ssl_control.train_peft_supervised_classifier",
        "scripts.experiments.central_ssl_control.train_peft_ssl_classifier",
        "scripts.datasets.run_dataset_pipeline",
        "scripts.datasets.materialize_query_ssl_split",
        "scripts.datasets.materialize_query_ssl_views",
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
