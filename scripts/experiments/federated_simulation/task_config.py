"""Federated simulation용 task/config 변환 helper."""

from __future__ import annotations

from main_server.src.services.federation.rounds.models import RoundOpenRequest
from shared.src.contracts.model_contracts import ModelManifest

from .models import FederatedTrainingTaskConfig


def build_round_open_request(
    *,
    active_manifest: ModelManifest,
    round_id: str,
    training_task_config: FederatedTrainingTaskConfig,
) -> RoundOpenRequest:
    """simulation task template을 canonical round open request로 변환한다."""
    return training_task_config.to_round_open_request(
        active_manifest=active_manifest,
        round_id=round_id,
    )
