"""Decision feedback signal construction for pseudo-label selection."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.contracts.training_contracts import (
    DecisionFeedbackSignal,
    TrainingTask,
)
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)


@dataclass(frozen=True, slots=True)
class PseudoLabelFeedbackBuilder:
    """Accepted candidate를 local feedback signal contract로 변환한다."""

    def build(
        self,
        *,
        candidate: PseudoLabelCandidate,
        training_task: TrainingTask,
    ) -> DecisionFeedbackSignal:
        return DecisionFeedbackSignal(
            schema_version="decision_feedback_signal.v1",
            signal_id=f"signal:{candidate.candidate_id}",
            signal_type="pseudo_label",
            label=candidate.label,
            confidence=candidate.confidence,
            occurred_at=candidate.occurred_at,
            source_event_ref=candidate.source_event_ref,
            task_context={
                "task_id": training_task.task_id,
                "round_id": training_task.round_id,
                "margin": candidate.margin,
                "runner_up_score": candidate.runner_up_score or 0.0,
                "confidence_kind": candidate.confidence_kind or "unknown",
                "sample_weight": candidate.sample_weight,
                "evidence_ref": candidate.evidence_ref or "",
            },
        )
