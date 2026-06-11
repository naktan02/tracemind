"""FL simulation용 agent scoring runtime bridge."""

from __future__ import annotations

from typing import Any


def build_federated_scoring_service(
    *,
    scoring_config: Any,
    similarity_name: str,
    shared_state: Any | None = None,
) -> Any:
    """agent ScoringService를 simulation 경계 뒤에서 조립한다."""

    from agent.src.services.inference.scoring_service import ScoringService

    return ScoringService.from_scoring_config(
        scoring_config,
        similarity_name=similarity_name,
        shared_state=shared_state,
    )
