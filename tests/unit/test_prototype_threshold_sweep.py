"""Prototype threshold policy unit tests."""

from __future__ import annotations

from scripts.experiments.prototype_strategy.evaluation import (
    evaluate_global_confidence_threshold,
)
from scripts.experiments.prototype_strategy.models import (
    PrototypeIndex,
    PrototypeVector,
    ScoredPrediction,
    ThresholdArtifact,
    ThresholdPolicyEvaluation,
)
from scripts.experiments.prototype_strategy.scoring import (
    PrototypeScoringConfig,
    build_prototype_index_scorer,
)
from scripts.experiments.prototype_strategy.sweep import (
    ThresholdPolicySelectionPolicy,
)
from scripts.experiments.prototype_strategy.threshold_policies import (
    ClasswiseStaticConfidencePolicy,
    FixMatchFixedConfidencePolicy,
    ValidationTargetErrorConfidencePolicy,
)


def test_evaluate_global_confidence_threshold_computes_accepted_metrics() -> None:
    predictions = (
        ScoredPrediction(
            actual_label="anxiety",
            predicted_label="anxiety",
            true_label_score=0.91,
            top1_score=0.91,
            top2_score=0.20,
            margin_top1_top2=0.71,
            is_correct=True,
        ),
        ScoredPrediction(
            actual_label="anxiety",
            predicted_label="depression",
            true_label_score=0.30,
            top1_score=0.82,
            top2_score=0.78,
            margin_top1_top2=0.04,
            is_correct=False,
        ),
        ScoredPrediction(
            actual_label="normal",
            predicted_label="normal",
            true_label_score=0.74,
            top1_score=0.74,
            top2_score=0.40,
            margin_top1_top2=0.34,
            is_correct=True,
        ),
    )

    metrics = evaluate_global_confidence_threshold(
        scored_predictions=predictions,
        categories=("anxiety", "depression", "normal"),
        confidence_threshold=0.8,
    )

    assert metrics.accepted_count == 2
    assert metrics.accepted_ratio == 2 / 3
    assert metrics.accepted_accuracy == 0.5
    assert metrics.accepted_correct_ratio == 1 / 3


def test_prototype_index_scorer_can_switch_to_top_k_mean_policy() -> None:
    scorer = build_prototype_index_scorer(
        scorer_backend_name="prototype_similarity",
        score_policy_name="topk_mean_cosine",
        score_top_k=2,
    )

    scores = scorer.score(
        [1.0, 0.0],
        PrototypeIndex(
            strategy_name="single",
            categories={
                "alert": [
                    PrototypeVector(
                        prototype_id="p1",
                        centroid=[1.0, 0.0],
                        member_count=1,
                    ),
                    PrototypeVector(
                        prototype_id="p2",
                        centroid=[0.0, 1.0],
                        member_count=1,
                    ),
                    PrototypeVector(
                        prototype_id="p3",
                        centroid=[-1.0, 0.0],
                        member_count=1,
                    ),
                ]
            },
        ),
    )

    assert scores["alert"] == 0.5


def test_prototype_index_scorer_accepts_canonical_scoring_config() -> None:
    scorer = build_prototype_index_scorer(
        config=PrototypeScoringConfig(
            scorer_backend_name="prototype_similarity",
            score_policy_name="topk_mean_cosine",
            score_top_k=2,
        )
    )

    scores = scorer.score(
        [1.0, 0.0],
        PrototypeIndex(
            strategy_name="single",
            categories={
                "alert": [
                    PrototypeVector(
                        prototype_id="p1",
                        centroid=[1.0, 0.0],
                        member_count=1,
                    ),
                    PrototypeVector(
                        prototype_id="p2",
                        centroid=[0.0, 1.0],
                        member_count=1,
                    ),
                    PrototypeVector(
                        prototype_id="p3",
                        centroid=[-1.0, 0.0],
                        member_count=1,
                    ),
                ]
            },
        ),
    )

    assert scores["alert"] == 0.5


def test_fixmatch_policy_builds_threshold_candidates() -> None:
    policy = FixMatchFixedConfidencePolicy(thresholds=(0.8, 0.95))
    predictions = (
        ScoredPrediction(
            actual_label="anxiety",
            predicted_label="anxiety",
            true_label_score=0.96,
            top1_score=0.96,
            top2_score=0.10,
            margin_top1_top2=0.86,
            is_correct=True,
        ),
        ScoredPrediction(
            actual_label="depression",
            predicted_label="depression",
            true_label_score=0.84,
            top1_score=0.84,
            top2_score=0.30,
            margin_top1_top2=0.54,
            is_correct=True,
        ),
    )

    evaluations = policy.build_evaluations(
        validation_predictions=predictions,
        test_predictions=predictions,
        categories=("anxiety", "depression"),
    )

    assert len(evaluations) == 2
    assert evaluations[0].policy_name == "fixmatch_fixed_confidence"
    assert (
        evaluations[0].threshold_artifact.parameters["confidence_threshold"] == 0.8
    )
    assert (
        evaluations[1].threshold_artifact.parameters["confidence_threshold"] == 0.95
    )


def test_target_error_policy_selects_maximum_coverage_feasible_threshold() -> None:
    policy = ValidationTargetErrorConfidencePolicy(target_errors=(0.1,))
    predictions = (
        ScoredPrediction(
            actual_label="anxiety",
            predicted_label="anxiety",
            true_label_score=0.95,
            top1_score=0.95,
            top2_score=0.10,
            margin_top1_top2=0.85,
            is_correct=True,
        ),
        ScoredPrediction(
            actual_label="depression",
            predicted_label="depression",
            true_label_score=0.92,
            top1_score=0.92,
            top2_score=0.30,
            margin_top1_top2=0.62,
            is_correct=True,
        ),
        ScoredPrediction(
            actual_label="normal",
            predicted_label="anxiety",
            true_label_score=0.12,
            top1_score=0.86,
            top2_score=0.10,
            margin_top1_top2=0.76,
            is_correct=False,
        ),
        ScoredPrediction(
            actual_label="anxiety",
            predicted_label="anxiety",
            true_label_score=0.80,
            top1_score=0.80,
            top2_score=0.40,
            margin_top1_top2=0.40,
            is_correct=True,
        ),
        ScoredPrediction(
            actual_label="depression",
            predicted_label="depression",
            true_label_score=0.70,
            top1_score=0.70,
            top2_score=0.45,
            margin_top1_top2=0.25,
            is_correct=True,
        ),
        ScoredPrediction(
            actual_label="normal",
            predicted_label="depression",
            true_label_score=0.20,
            top1_score=0.65,
            top2_score=0.30,
            margin_top1_top2=0.35,
            is_correct=False,
        ),
    )

    evaluations = policy.build_evaluations(
        validation_predictions=predictions,
        test_predictions=predictions,
        categories=("anxiety", "depression", "normal"),
    )

    assert len(evaluations) == 1
    assert evaluations[0].policy_name == "validation_target_error_confidence"
    assert evaluations[0].selection_params["target_error"] == 0.1
    assert evaluations[0].threshold_artifact.parameters["confidence_threshold"] == 0.92
    assert evaluations[0].validation_metrics.accepted_accuracy == 1.0
    assert evaluations[0].validation_metrics.accepted_ratio == 2 / 6


def test_classwise_static_policy_builds_label_specific_thresholds() -> None:
    policy = ClasswiseStaticConfidencePolicy(target_errors=(0.1,))
    predictions = (
        ScoredPrediction(
            actual_label="anxiety",
            predicted_label="anxiety",
            true_label_score=0.95,
            top1_score=0.95,
            top2_score=0.10,
            margin_top1_top2=0.85,
            is_correct=True,
        ),
        ScoredPrediction(
            actual_label="normal",
            predicted_label="anxiety",
            true_label_score=0.20,
            top1_score=0.70,
            top2_score=0.10,
            margin_top1_top2=0.60,
            is_correct=False,
        ),
        ScoredPrediction(
            actual_label="depression",
            predicted_label="depression",
            true_label_score=0.92,
            top1_score=0.92,
            top2_score=0.20,
            margin_top1_top2=0.72,
            is_correct=True,
        ),
        ScoredPrediction(
            actual_label="depression",
            predicted_label="depression",
            true_label_score=0.82,
            top1_score=0.82,
            top2_score=0.30,
            margin_top1_top2=0.52,
            is_correct=True,
        ),
        ScoredPrediction(
            actual_label="normal",
            predicted_label="normal",
            true_label_score=0.88,
            top1_score=0.88,
            top2_score=0.20,
            margin_top1_top2=0.68,
            is_correct=True,
        ),
    )

    evaluations = policy.build_evaluations(
        validation_predictions=predictions,
        test_predictions=predictions,
        categories=("anxiety", "depression", "normal"),
    )

    assert len(evaluations) == 1
    artifact = evaluations[0].threshold_artifact
    assert artifact.threshold_kind == "classwise_confidence"
    assert artifact.parameters["confidence_threshold_by_label"]["anxiety"] == 0.95
    assert artifact.parameters["confidence_threshold_by_label"]["depression"] == 0.82
    assert artifact.parameters["confidence_threshold_by_label"]["normal"] == 0.88
    assert evaluations[0].validation_metrics.accepted_accuracy == 1.0


def test_threshold_policy_selection_prefers_precision_once_coverage_floor_is_met(
) -> None:
    policy = ThresholdPolicySelectionPolicy(minimum_accepted_ratio=0.5)

    def evaluation(
        *,
        policy_name: str,
        confidence_threshold: float,
        accepted_correct_ratio: float,
        accepted_accuracy: float,
        accepted_ratio: float,
    ) -> ThresholdPolicyEvaluation:
        metrics = evaluate_global_confidence_threshold(
            scored_predictions=(),
            categories=(),
            confidence_threshold=confidence_threshold,
        )
        metrics.accepted_correct_ratio = accepted_correct_ratio
        metrics.accepted_accuracy = accepted_accuracy
        metrics.accepted_ratio = accepted_ratio
        return ThresholdPolicyEvaluation(
            policy_name=policy_name,
            candidate_name=f"confidence={confidence_threshold:.2f}",
            source_paper=FixMatchFixedConfidencePolicy().source_paper,
            selection_params={"confidence_threshold": confidence_threshold},
            threshold_artifact=ThresholdArtifact(
                threshold_kind="global_confidence",
                parameters={"confidence_threshold": confidence_threshold},
            ),
            validation_metrics=metrics,
            test_metrics=metrics,
        )

    selected = policy.select(
        [
            evaluation(
                policy_name="fixmatch_fixed_confidence",
                confidence_threshold=0.8,
                accepted_correct_ratio=0.01,
                accepted_accuracy=1.0,
                accepted_ratio=0.01,
            ),
            evaluation(
                policy_name="validation_target_error_confidence",
                confidence_threshold=0.7,
                accepted_correct_ratio=0.44,
                accepted_accuracy=0.88,
                accepted_ratio=0.50,
            ),
            evaluation(
                policy_name="fixmatch_fixed_confidence",
                confidence_threshold=0.6,
                accepted_correct_ratio=0.69,
                accepted_accuracy=0.75,
                accepted_ratio=0.92,
            ),
        ]
    )

    assert selected.policy_name == "validation_target_error_confidence"
    assert selected.threshold_artifact.parameters["confidence_threshold"] == 0.7
