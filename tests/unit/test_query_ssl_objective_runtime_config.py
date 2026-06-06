"""Query SSL objective runtime config tests."""

from __future__ import annotations

import pytest

from methods.ssl.runtime.objective_config import QuerySslObjectiveRuntimeConfig
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


def test_query_ssl_objective_config_reads_algorithm_parameters_from_extras() -> None:
    objective = TrainingObjectiveConfig(
        training_backend_name="peft_classifier_trainer",
        extras={
            "query_ssl.method_name": "fixmatch_usb_v1",
            "query_ssl.algorithm_name": "fixmatch",
            "query_ssl.strong_view_policy": "first_aug",
            "query_ssl.unlabeled_batch_size": 8,
            "query_ssl.temperature": 0.5,
            "query_ssl.p_cutoff": 0.95,
        },
    )

    config = QuerySslObjectiveRuntimeConfig.from_objective_config(objective)

    assert config is not None
    assert config.method_name == "fixmatch_usb_v1"
    assert config.algorithm_name == "fixmatch"
    assert config.strong_view_policy == "first_aug"
    assert config.unlabeled_batch_size == 8
    assert config.parameters == {
        "unlabeled_batch_size": 8,
        "temperature": 0.5,
        "p_cutoff": 0.95,
    }


def test_query_ssl_objective_config_requires_method_and_algorithm_together() -> None:
    objective = TrainingObjectiveConfig(
        training_backend_name="peft_classifier_trainer",
        extras={"query_ssl.algorithm_name": "fixmatch"},
    )

    with pytest.raises(ValueError, match="require both method_name and algorithm_name"):
        QuerySslObjectiveRuntimeConfig.from_objective_config(objective)
