"""Pseudo-label candidate construction helpers."""

from __future__ import annotations

from dataclasses import dataclass

from methods.ssl.hooks.selection import (
    PseudoLabelSelectionConfig,
    PseudoLabelSelectionHook,
)
from shared.src.contracts.training_contracts import TrainingTask
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


@dataclass(frozen=True, slots=True)
class SelectionContextSeed:
    """후보 생성 시점에 고정해야 하는 selection provenance."""

    pseudo_label_algorithm_name: str
    evidence_backend_name: str
    evidence_view_kind: str


@dataclass(frozen=True, slots=True)
class BuiltPseudoLabelCandidate:
    """Hook decision과 context seed가 결합된 후보."""

    candidate: PseudoLabelCandidate
    context_seed: SelectionContextSeed


@dataclass(frozen=True, slots=True)
class PseudoLabelCandidateBuilder:
    """Evidence와 SSL selection hook decision을 candidate contract로 변환한다."""

    default_evidence_backend_name: str

    def build(
        self,
        *,
        evidence: PseudoLabelEvidence,
        training_task: TrainingTask,
        selection_config: PseudoLabelSelectionConfig,
        selection_hook: PseudoLabelSelectionHook,
    ) -> BuiltPseudoLabelCandidate:
        decision = selection_hook.evaluate(
            evidence=evidence,
            config=selection_config,
        )

        return BuiltPseudoLabelCandidate(
            candidate=PseudoLabelCandidate(
                schema_version="pseudo_label_candidate.v1",
                candidate_id=(f"{training_task.round_id}:{evidence.source_event_ref}"),
                source_event_ref=evidence.source_event_ref,
                occurred_at=evidence.occurred_at,
                label=decision.label,
                confidence=decision.confidence,
                margin=decision.margin,
                accepted=decision.accepted,
                runner_up_label=decision.runner_up_label,
                runner_up_score=decision.runner_up_score,
                evidence_ref=evidence.evidence_id,
                sample_weight=decision.sample_weight,
                task_id=training_task.task_id,
                round_id=training_task.round_id,
            ),
            context_seed=SelectionContextSeed(
                pseudo_label_algorithm_name=selection_hook.hook_name,
                evidence_backend_name=self._resolve_evidence_backend_name(
                    evidence=evidence,
                    training_task=training_task,
                ),
                evidence_view_kind=evidence.view_kind,
            ),
        )

    def _resolve_evidence_backend_name(
        self,
        *,
        evidence: PseudoLabelEvidence,
        training_task: TrainingTask,
    ) -> str:
        evidence_backend_name = evidence.metadata.get("evidence_backend_name")
        if isinstance(evidence_backend_name, str) and evidence_backend_name.strip():
            return evidence_backend_name
        return (
            training_task.objective_config.evidence_backend_name
            or self.default_evidence_backend_name
        )
