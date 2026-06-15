"""Fixed-feature classifier 학습과 예측 core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import softmax

from methods.classification.fixed_feature.estimators import build_estimator
from methods.classification.fixed_feature.evaluation import (
    evaluate_fixed_feature_predictions,
)
from methods.classification.fixed_feature.feature_spaces import build_feature_space


@dataclass(frozen=True, slots=True)
class FixedFeatureDataset:
    """고정 feature 분류 core가 받는 text/label 데이터."""

    texts: list[str]
    labels: list[str]


@dataclass(frozen=True, slots=True)
class FixedFeatureEvaluation:
    """한 eval split의 예측과 metric."""

    report: dict[str, Any]
    predicted_labels: list[str]
    score_matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class FixedFeatureTrainingResult:
    """학습된 fixed-feature classifier와 eval 결과."""

    feature_space: Any
    estimator: Any
    evaluations: dict[str, FixedFeatureEvaluation]


def run_fixed_feature_classification(
    *,
    feature_space_config: Mapping[str, Any],
    estimator_config: Mapping[str, Any],
    categories: list[str],
    train_dataset: FixedFeatureDataset,
    eval_datasets: Mapping[str, FixedFeatureDataset],
) -> FixedFeatureTrainingResult:
    """feature space와 estimator를 학습하고 모든 eval split을 평가한다."""

    _validate_labels(categories=categories, labels=train_dataset.labels)
    feature_space = build_feature_space(feature_space_config)
    estimator = build_estimator(estimator_config)
    train_features = feature_space.fit_transform(train_dataset.texts)
    estimator.fit(train_features, train_dataset.labels)

    evaluations: dict[str, FixedFeatureEvaluation] = {}
    for dataset_name, dataset in eval_datasets.items():
        _validate_labels(categories=categories, labels=dataset.labels)
        features = feature_space.transform(dataset.texts)
        predicted_labels = [str(label) for label in estimator.predict(features)]
        score_matrix = _predict_score_matrix(
            estimator=estimator,
            features=features,
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

    return FixedFeatureTrainingResult(
        feature_space=feature_space,
        estimator=estimator,
        evaluations=evaluations,
    )


def _predict_score_matrix(
    *,
    estimator: Any,
    features: Any,
    categories: list[str],
) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        raw_scores = np.asarray(estimator.predict_proba(features), dtype=float)
        return _align_class_columns(
            raw_scores=raw_scores,
            estimator_classes=[str(label) for label in estimator.classes_],
            categories=categories,
            normalize=True,
        )
    if hasattr(estimator, "decision_function"):
        raw_scores = np.asarray(estimator.decision_function(features), dtype=float)
        if raw_scores.ndim == 1:
            raw_scores = np.column_stack([-raw_scores, raw_scores])
        aligned = _align_class_columns(
            raw_scores=raw_scores,
            estimator_classes=[str(label) for label in estimator.classes_],
            categories=categories,
            normalize=False,
        )
        return softmax(aligned, axis=1)
    predicted_labels = [str(label) for label in estimator.predict(features)]
    label_to_index = {label: index for index, label in enumerate(categories)}
    scores = np.zeros((len(predicted_labels), len(categories)), dtype=float)
    for row_index, label in enumerate(predicted_labels):
        scores[row_index, label_to_index[label]] = 1.0
    return scores


def _align_class_columns(
    *,
    raw_scores: np.ndarray,
    estimator_classes: Sequence[str],
    categories: list[str],
    normalize: bool,
) -> np.ndarray:
    class_to_raw_index = {label: index for index, label in enumerate(estimator_classes)}
    aligned = np.zeros((raw_scores.shape[0], len(categories)), dtype=float)
    for category_index, category in enumerate(categories):
        raw_index = class_to_raw_index.get(category)
        if raw_index is not None:
            aligned[:, category_index] = raw_scores[:, raw_index]
    if not normalize:
        return aligned
    row_sums = aligned.sum(axis=1, keepdims=True)
    np.divide(aligned, row_sums, out=aligned, where=row_sums > 0)
    return aligned


def _validate_labels(*, categories: list[str], labels: Sequence[str]) -> None:
    allowed = set(categories)
    unknown_labels = sorted({label for label in labels if label not in allowed})
    if unknown_labels:
        raise ValueError(
            f"Unknown labels for fixed-feature classification: {unknown_labels}"
        )
