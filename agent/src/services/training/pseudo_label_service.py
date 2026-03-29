"""Pseudo-label 후보 생성과 필터링 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.decision_feedback_signal import (
    DecisionFeedbackSignal,
)
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)
from shared.src.domain.entities.training.training_task import TrainingTask


@dataclass(slots=True)
class PseudoLabelSelectionConfig:
    """pseudo-label 선택 기본 설정."""

    confidence_threshold: float = 0.8
    margin_threshold: float = 0.15


@dataclass(slots=True)
class PseudoLabelSelectionResult:
    """pseudo-label 후보와 최종 채택 결과."""

    candidates: tuple[PseudoLabelCandidate, ...]
    accepted_candidates: tuple[PseudoLabelCandidate, ...]
    feedback_signals: tuple[DecisionFeedbackSignal, ...]

    @property
    def total_count(self) -> int:
        return len(self.candidates)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_candidates)

    @property
    def accepted_ratio(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.accepted_count / self.total_count


@dataclass(slots=True)
class PseudoLabelSelectionService:
    """score threshold와 margin 기준으로 pseudo-label을 선별한다."""

    config: PseudoLabelSelectionConfig = field(
        default_factory=PseudoLabelSelectionConfig
    )

    def select(
        self,
        *,
        scored_events: tuple[ScoredEvent, ...] | list[ScoredEvent],
        training_task: TrainingTask,
    ) -> PseudoLabelSelectionResult:
        confidence_threshold = self._get_float(
            training_task.objective_config,
            key="confidence_threshold",
            default=self.config.confidence_threshold,
        )
        margin_threshold = self._get_float(
            training_task.objective_config,
            key="margin_threshold",
            default=self.config.margin_threshold,
        )
        max_examples = self._get_int(
            training_task.selection_policy,
            key="max_examples",
            default=None,
        )

        initial_candidates = [
            self._build_candidate(
                scored_event=event,
                training_task=training_task,
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
            )
            for event in scored_events
        ]
        prelim_accepted = [
            candidate for candidate in initial_candidates if candidate.accepted
        ]
        prelim_accepted.sort(
            key=lambda candidate: (
                -candidate.confidence,
                -candidate.margin,
                candidate.source_event_ref,
            )
        )

        if max_examples is not None:
            selected_ids = {
                candidate.candidate_id
                for candidate in prelim_accepted[: max(max_examples, 0)]
            }
        else:
            selected_ids = {candidate.candidate_id for candidate in prelim_accepted}

        finalized_candidates: list[PseudoLabelCandidate] = []
        accepted_candidates: list[PseudoLabelCandidate] = []
        feedback_signals: list[DecisionFeedbackSignal] = []
        for candidate in initial_candidates:
            is_selected = candidate.candidate_id in selected_ids
            finalized = candidate if is_selected else replace(candidate, accepted=False)
            finalized_candidates.append(finalized)
            if finalized.accepted:
                accepted_candidates.append(finalized)
                feedback_signals.append(
                    self._to_feedback_signal(
                        candidate=finalized,
                        training_task=training_task,
                    )
                )

        return PseudoLabelSelectionResult(
            candidates=tuple(finalized_candidates),
            accepted_candidates=tuple(accepted_candidates),
            feedback_signals=tuple(feedback_signals),
        )

    def _build_candidate(
        self,
        *,
        scored_event: ScoredEvent,
        training_task: TrainingTask,
        confidence_threshold: float,
        margin_threshold: float,
    ) -> PseudoLabelCandidate:
        ranked_scores = sorted(
            scored_event.category_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked_scores:
            raise ValueError("ScoredEvent must contain at least one category score.")

        top_label, top_score = ranked_scores[0]
        if len(ranked_scores) > 1:
            runner_up_label, runner_up_score = ranked_scores[1]
        else:
            runner_up_label, runner_up_score = None, 0.0
        margin = top_score - runner_up_score
        accepted = top_score >= confidence_threshold and margin >= margin_threshold

        return PseudoLabelCandidate(
            schema_version="pseudo_label_candidate.v1",
            candidate_id=f"{training_task.round_id}:{scored_event.query_id}",
            source_event_ref=scored_event.query_id,
            occurred_at=scored_event.occurred_at,
            label=top_label,
            confidence=top_score,
            margin=margin,
            accepted=accepted,
            runner_up_label=runner_up_label,
            runner_up_score=runner_up_score,
            task_id=training_task.task_id,
            round_id=training_task.round_id,
            metadata={
                "confidence_threshold": confidence_threshold,
                "margin_threshold": margin_threshold,
            },
        )

    @staticmethod
    def _to_feedback_signal(
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
            },
        )

    @staticmethod
    def _get_float(
        source: dict[str, str | int | float | bool],
        *,
        key: str,
        default: float,
    ) -> float:
        value = source.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"Expected float-like value for '{key}', got bool.")
        return float(value)

    @staticmethod
    def _get_int(
        source: dict[str, str | int | float | bool],
        *,
        key: str,
        default: int | None,
    ) -> int | None:
        value = source.get(key, default)
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"Expected int-like value for '{key}', got bool.")
        return int(value)
