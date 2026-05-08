"""FL simulation용 agent pseudo-label selection bridge."""

from __future__ import annotations

from typing import Any


def select_federated_pseudo_labels(
    *,
    scored_events: tuple[Any, ...],
    training_task: Any,
) -> Any:
    """validation pseudo-label selection을 agent policy로 실행한다."""

    from agent.src.services.training.selection.pseudo_label_service import (
        PseudoLabelSelectionService,
    )

    return PseudoLabelSelectionService().select(
        scored_events=scored_events,
        training_task=training_task,
    )
