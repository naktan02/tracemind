"""Query SSL algorithm report state summary tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest
import torch

from methods.ssl.algorithms.dash.dash import DashAlgorithm
from methods.ssl.algorithms.softmatch.softmatch import SoftMatchAlgorithm
from methods.ssl.state import (
    build_query_ssl_algorithm_state,
    export_query_ssl_algorithm_report_state_summary,
)


class _TensorStateAlgorithm:
    algorithm_name = "tensor_state"

    def export_state(self) -> Mapping[str, Any]:
        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=True,
            metadata={"step": 3},
            tensors={
                "class_distribution": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                "scalar_threshold": torch.tensor(0.75),
            },
        )


def test_report_state_summary_converts_tensor_state_to_json_safe_summary() -> None:
    summary = dict(
        export_query_ssl_algorithm_report_state_summary(_TensorStateAlgorithm())
    )

    json.dumps(summary)
    assert summary["step"] == 3
    assert summary["scalar_threshold"] == 0.75
    tensor_summary = summary["class_distribution"]
    assert isinstance(tensor_summary, dict)
    assert tensor_summary == {
        "type": "tensor",
        "shape": [4],
        "dtype": "torch.float32",
        "numel": 4,
        "mean": tensor_summary["mean"],
        "std": tensor_summary["std"],
        "min": tensor_summary["min"],
        "max": tensor_summary["max"],
    }
    assert tensor_summary["mean"] == pytest.approx(0.25)
    assert tensor_summary["std"] == pytest.approx(0.1118034)
    assert tensor_summary["min"] == pytest.approx(0.1)
    assert tensor_summary["max"] == pytest.approx(0.4)


def test_dash_report_state_summary_exports_method_specific_scalars() -> None:
    algorithm = DashAlgorithm(num_wu_iter=1024)
    algorithm.configure_initial_selection_loss(selection_loss=1.25)
    algorithm.thresholding_hook.state.rho = 0.8
    algorithm.thresholding_hook.state.rho_update_cnt = 2

    summary = dict(export_query_ssl_algorithm_report_state_summary(algorithm))

    json.dumps(summary)
    assert summary == {
        "rho_init": 1.25,
        "rho": 0.8,
        "rho_update_cnt": 2,
        "use_hard_label": False,
        "num_wu_iter": 1024,
        "rho_init_source": "supervised_warmup_selection_loss",
    }


def test_softmatch_report_state_summary_is_json_safe_after_dataset_config() -> None:
    algorithm = SoftMatchAlgorithm(temperature=0.5)
    algorithm.configure_dataset(num_classes=4, unlabeled_row_count=16)

    summary = dict(export_query_ssl_algorithm_report_state_summary(algorithm))

    json.dumps(summary)
    assert summary["algorithm_name"] == "softmatch"
    assert summary["num_classes"] == 4
    assert isinstance(summary["prob_max_mu_t"], float)
    assert isinstance(summary["prob_max_var_t"], float)
    assert summary["p_target"]["shape"] == [4]
