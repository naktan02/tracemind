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
        "scripts.experiments.central.ssl_control.run_peft_supervised_control",
        "scripts.experiments.central.ssl_control.run_peft_ssl_control",
        "scripts.workflows.datasets.run_dataset_pipeline",
        "scripts.workflows.datasets.materialize_query_ssl_split",
        "scripts.workflows.datasets.materialize_query_ssl_views",
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
