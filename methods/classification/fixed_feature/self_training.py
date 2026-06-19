"""Fixed-feature self-training classification core."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier

from methods.classification.fixed_feature.estimators import build_estimator
from methods.classification.fixed_feature.evaluation import (
    evaluate_fixed_feature_predictions,
)
from methods.classification.fixed_feature.feature_spaces import build_feature_space
from methods.classification.fixed_feature.training import (
    FixedFeatureDataset,
    FixedFeatureEvaluation,
    _predict_score_matrix,
    _validate_feature_estimator_pair,
    _validate_labels,
)


@dataclass(frozen=True, slots=True)
class FixedFeatureUnlabeledDataset:
    """Self-training core가 받는 unlabeled text 데이터."""

    texts: list[str]
    query_ids: list[str] | None = None


@dataclass(frozen=True, slots=True)
class FixedFeaturePseudoLabelRecord:
    """unlabeled row별 self-training pseudo-label 결과."""

    query_id: str | None
    predicted_label: str | None
    accepted: bool
    iteration: int | None
    score_by_category: dict[str, float]


@dataclass(frozen=True, slots=True)
class FixedFeatureSelfTrainingSummary:
    """Self-training pseudo-label 채택 요약."""

    labeled_count: int
    unlabeled_count: int
    accepted_pseudo_label_count: int
    accepted_class_distribution: dict[str, int]
    n_iter: int
    termination_condition: str
    criterion: str
    threshold: float
    k_best: int
    max_iter: int
    unlabeled_label: int


@dataclass(frozen=True, slots=True)
class FixedFeatureSelfTrainingResult:
    """학습된 fixed-feature self-training classifier와 eval 결과."""

    feature_space: Any
    estimator: SelfTrainingClassifier
    evaluations: dict[str, FixedFeatureEvaluation]
    pseudo_label_records: list[FixedFeaturePseudoLabelRecord]
    summary: FixedFeatureSelfTrainingSummary


def run_fixed_feature_self_training_classification(
    *,
    feature_space_config: Mapping[str, Any],
    estimator_config: Mapping[str, Any],
    self_training_config: Mapping[str, Any],
    categories: list[str],
    train_dataset: FixedFeatureDataset,
    unlabeled_dataset: FixedFeatureUnlabeledDataset,
    eval_datasets: Mapping[str, FixedFeatureDataset],
) -> FixedFeatureSelfTrainingResult:
    """SelfTrainingClassifier로 fixed-feature 준지도 baseline을 학습/평가한다."""

    _validate_labels(categories=categories, labels=train_dataset.labels)
    _validate_feature_estimator_pair(
        feature_space_config=feature_space_config,
        estimator_config=estimator_config,
    )
    if len(unlabeled_dataset.query_ids or []) not in {0, len(unlabeled_dataset.texts)}:
        raise ValueError("unlabeled query_ids length must match unlabeled texts.")

    feature_space = build_feature_space(feature_space_config)
    base_estimator = build_estimator(estimator_config)
    if not hasattr(base_estimator, "predict_proba"):
        raise ValueError(
            "fixed-feature self-training requires a base estimator with "
            f"predict_proba; unsupported estimator={estimator_config.get('name')!r}."
        )

    unlabeled_label = int(self_training_config.get("unlabeled_label", -1))
    combined_texts = [*train_dataset.texts, *unlabeled_dataset.texts]
    combined_labels = np.asarray(
        [*train_dataset.labels, *([unlabeled_label] * len(unlabeled_dataset.texts))],
        dtype=object,
    )
    features = feature_space.fit_transform(combined_texts)
    estimator = SelfTrainingClassifier(
        estimator=base_estimator,
        threshold=float(self_training_config.get("threshold", 0.75)),
        criterion=str(self_training_config.get("criterion", "threshold")),
        k_best=int(self_training_config.get("k_best", 10)),
        max_iter=int(self_training_config.get("max_iter", 10)),
        verbose=bool(self_training_config.get("verbose", False)),
    )
    estimator.fit(features, combined_labels)

    evaluations: dict[str, FixedFeatureEvaluation] = {}
    for dataset_name, dataset in eval_datasets.items():
        _validate_labels(categories=categories, labels=dataset.labels)
        eval_features = feature_space.transform(dataset.texts)
        predicted_labels = [str(label) for label in estimator.predict(eval_features)]
        score_matrix = _predict_score_matrix(
            estimator=estimator,
            features=eval_features,
            categories=categories,
        )
        evaluations[str(dataset_name)] = FixedFeatureEvaluation(
            report=evaluate_fixed_feature_predictions(
                categories=categories,
                actual_labels=dataset.labels,
                predicted_labels=predicted_labels,
                score_matrix=score_matrix,
            ),
            predicted_labels=predicted_labels,
            score_matrix=score_matrix,
        )

    unlabeled_features = feature_space.transform(unlabeled_dataset.texts)
    unlabeled_scores = _predict_score_matrix(
        estimator=estimator,
        features=unlabeled_features,
        categories=categories,
    )
    pseudo_label_records = _build_pseudo_label_records(
        estimator=estimator,
        categories=categories,
        unlabeled_dataset=unlabeled_dataset,
        score_matrix=unlabeled_scores,
        labeled_count=len(train_dataset.texts),
    )
    summary = _build_self_training_summary(
        records=pseudo_label_records,
        labeled_count=len(train_dataset.texts),
        unlabeled_count=len(unlabeled_dataset.texts),
        estimator=estimator,
        self_training_config=self_training_config,
        unlabeled_label=unlabeled_label,
    )

    return FixedFeatureSelfTrainingResult(
        feature_space=feature_space,
        estimator=estimator,
        evaluations=evaluations,
        pseudo_label_records=pseudo_label_records,
        summary=summary,
    )


def _build_pseudo_label_records(
    *,
    estimator: SelfTrainingClassifier,
    categories: list[str],
    unlabeled_dataset: FixedFeatureUnlabeledDataset,
    score_matrix: np.ndarray,
    labeled_count: int,
) -> list[FixedFeaturePseudoLabelRecord]:
    query_ids = unlabeled_dataset.query_ids or [None] * len(unlabeled_dataset.texts)
    transduction = list(estimator.transduction_[labeled_count:])
    labeled_iter = list(estimator.labeled_iter_[labeled_count:])
    records: list[FixedFeaturePseudoLabelRecord] = []
    for query_id, label, iteration, scores in zip(
        query_ids,
        transduction,
        labeled_iter,
        score_matrix,
        strict=True,
    ):
        iteration_int = int(iteration)
        accepted = iteration_int > 0
        records.append(
            FixedFeaturePseudoLabelRecord(
                query_id=None if query_id is None else str(query_id),
                predicted_label=str(label) if accepted else None,
                accepted=accepted,
                iteration=iteration_int if accepted else None,
                score_by_category={
                    category: round(float(score), 6)
                    for category, score in zip(categories, scores, strict=True)
                },
            )
        )
    return records


def _build_self_training_summary(
    *,
    records: list[FixedFeaturePseudoLabelRecord],
    labeled_count: int,
    unlabeled_count: int,
    estimator: SelfTrainingClassifier,
    self_training_config: Mapping[str, Any],
    unlabeled_label: int,
) -> FixedFeatureSelfTrainingSummary:
    accepted_labels = [
        record.predicted_label
        for record in records
        if record.accepted and record.predicted_label is not None
    ]
    return FixedFeatureSelfTrainingSummary(
        labeled_count=labeled_count,
        unlabeled_count=unlabeled_count,
        accepted_pseudo_label_count=len(accepted_labels),
        accepted_class_distribution=dict(sorted(Counter(accepted_labels).items())),
        n_iter=int(estimator.n_iter_),
        termination_condition=str(estimator.termination_condition_),
        criterion=str(self_training_config.get("criterion", "threshold")),
        threshold=float(self_training_config.get("threshold", 0.75)),
        k_best=int(self_training_config.get("k_best", 10)),
        max_iter=int(self_training_config.get("max_iter", 10)),
        unlabeled_label=unlabeled_label,
    )
