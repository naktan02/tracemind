"""Pseudo-label selection pipeline unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.src.services.training.selection.candidate_builder import (
    BuiltPseudoLabelCandidate,
    PseudoLabelCandidateBuilder,
    SelectionContextSeed,
)
from agent.src.services.training.selection.cap_policy import PseudoLabelCapPolicy
from agent.src.services.training.selection.selector import PseudoLabelSelector
from methods.ssl.hooks.selection import (
    MarginThresholdPseudoLabelSelectionHook,
    PseudoLabelSelectionConfig,
)
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
    PseudoLabelSelectionStage,
)
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


def test_candidate_builder_preserves_hook_decision_and_context_seed() -> None:
    builder = PseudoLabelCandidateBuilder(
        default_evidence_backend_name="default_evidence_backend"
    )
    evidence = _build_evidence(
        source_event_ref="q1",
        top1_score=0.91,
        margin=0.2,
        metadata={"evidence_backend_name": "metadata_backend"},
    )

    built = builder.build(
        evidence=evidence,
        training_task=_build_task(),
        selection_config=PseudoLabelSelectionConfig(
            parameters={
                "confidence_threshold": 0.6,
                "margin_threshold": 0.05,
            },
        ),
        selection_hook=MarginThresholdPseudoLabelSelectionHook(),
    )

    assert built.candidate.candidate_id == "round_0001:q1"
    assert built.candidate.accepted is True
    assert built.candidate.label == "anxiety"
    assert built.candidate.evidence_ref == "evidence:q1"
    assert built.context_seed.pseudo_label_algorithm_name == "top1_margin_threshold"
    assert built.context_seed.evidence_backend_name == "metadata_backend"
    assert built.context_seed.evidence_view_kind == "single_view"


def test_cap_policy_ranks_by_confidence_margin_and_source_ref() -> None:
    decision = PseudoLabelCapPolicy().decide(
        candidates=(
            _build_candidate("round:q_low", confidence=0.7, margin=0.4),
            _build_candidate("round:q_high", confidence=0.9, margin=0.1),
            _build_candidate("round:q_tie_a", confidence=0.8, margin=0.2),
            _build_candidate("round:q_tie_b", confidence=0.8, margin=0.2),
            _build_candidate(
                "round:q_rejected",
                confidence=0.99,
                margin=0.9,
                accepted=False,
            ),
        ),
        max_examples=3,
    )

    assert decision.selected_candidate_ids == frozenset(
        {"round:q_high", "round:q_tie_a", "round:q_tie_b"}
    )
    assert decision.pre_cap_ranks == {
        "round:q_high": 1,
        "round:q_tie_a": 2,
        "round:q_tie_b": 3,
        "round:q_low": 4,
    }


def test_selector_finalizes_context_metadata_and_feedback_after_cap() -> None:
    selector = PseudoLabelSelector()
    built_candidates = (
        _build_built_candidate(
            _build_candidate("round:q_keep", confidence=0.9, margin=0.2)
        ),
        _build_built_candidate(
            _build_candidate("round:q_drop", confidence=0.8, margin=0.2)
        ),
        _build_built_candidate(
            _build_candidate(
                "round:q_threshold",
                confidence=0.95,
                margin=0.01,
                accepted=False,
            )
        ),
    )

    result = selector.finalize(
        built_candidates=built_candidates,
        training_task=_build_task(),
        selection_config=PseudoLabelSelectionConfig(
            parameters={
                "confidence_threshold": 0.6,
                "margin_threshold": 0.05,
            },
        ),
        max_examples=1,
    )

    by_id = {candidate.candidate_id: candidate for candidate in result.candidates}
    accepted_candidate_ids = tuple(
        candidate.candidate_id for candidate in result.accepted_candidates
    )
    assert accepted_candidate_ids == ("round:q_keep",)
    assert len(result.feedback_signals) == 1
    assert result.feedback_signals[0].signal_id == "signal:round:q_keep"
    assert by_id["round:q_keep"].selection_context is not None
    assert by_id["round:q_keep"].selection_context.selection_stage == (
        PseudoLabelSelectionStage.ACCEPTED
    )
    assert by_id["round:q_keep"].metadata["pre_cap_rank"] == 1
    assert by_id["round:q_drop"].selection_context is not None
    assert by_id["round:q_drop"].selection_context.selection_stage == (
        PseudoLabelSelectionStage.DROPPED_BY_CAP
    )
    assert by_id["round:q_drop"].metadata["selected_by_cap"] is False
    assert by_id["round:q_threshold"].selection_context is not None
    assert by_id["round:q_threshold"].selection_context.selection_stage == (
        PseudoLabelSelectionStage.POLICY_REJECTED
    )
    assert "pre_cap_rank" not in by_id["round:q_threshold"].metadata


def _build_task() -> TrainingTask:
    return TrainingTask(
        schema_version="training_task.v1",
        task_id="task_001",
        round_id="round_0001",
        model_id="tracemind-embed",
        model_revision="rev_000",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=10,
        objective_config=TrainingObjectiveConfig(
            training_backend_name="peft_classifier_trainer",
            extras={
                "selection.confidence_threshold": 0.6,
                "selection.margin_threshold": 0.05,
            },
        ),
        selection_policy=TrainingSelectionPolicy(max_examples=1),
    )


def _build_evidence(
    *,
    source_event_ref: str,
    top1_score: float,
    margin: float,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> PseudoLabelEvidence:
    return PseudoLabelEvidence(
        schema_version="pseudo_label_evidence.v1",
        evidence_id=f"evidence:{source_event_ref}",
        source_event_ref=source_event_ref,
        occurred_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        label="anxiety",
        confidence=top1_score,
        margin=margin,
        top1_label="anxiety",
        top1_score=top1_score,
        top2_label="normal",
        top2_score=top1_score - margin,
        sample_weight=top1_score,
        metadata=metadata or {},
    )


def _build_candidate(
    candidate_id: str,
    *,
    confidence: float,
    margin: float,
    accepted: bool = True,
) -> PseudoLabelCandidate:
    source_event_ref = candidate_id.split(":", maxsplit=1)[1]
    return PseudoLabelCandidate(
        schema_version="pseudo_label_candidate.v1",
        candidate_id=candidate_id,
        source_event_ref=source_event_ref,
        occurred_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        label="anxiety",
        confidence=confidence,
        margin=margin,
        accepted=accepted,
        runner_up_label="normal",
        runner_up_score=confidence - margin,
        evidence_ref=f"evidence:{source_event_ref}",
        sample_weight=confidence,
        task_id="task_001",
        round_id="round_0001",
    )


def _build_built_candidate(
    candidate: PseudoLabelCandidate,
) -> BuiltPseudoLabelCandidate:
    return BuiltPseudoLabelCandidate(
        candidate=candidate,
        context_seed=SelectionContextSeed(
            pseudo_label_algorithm_name="top1_margin_threshold",
            evidence_backend_name="analysis_score_evidence",
            evidence_view_kind="single_view",
        ),
    )
