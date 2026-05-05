"""Federated simulation용 task/config 변환 helper."""

from __future__ import annotations

from typing import Any

from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedTrainingTaskConfig,
)
from scripts.runtime_adapters.federated_server_runtime import (
    build_round_open_request as build_server_round_open_request,
)
from shared.src.contracts.model_contracts import ModelManifest


def build_round_open_request(
    *,
    active_manifest: ModelManifest,
    round_id: str,
    training_task_config: FederatedTrainingTaskConfig,
) -> Any:
    """simulation task template을 canonical round open request로 변환한다."""
    return build_server_round_open_request(
        active_manifest=active_manifest,
        round_id=round_id,
        training_task_config=training_task_config,
    )
