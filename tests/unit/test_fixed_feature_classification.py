"""fixed-feature classification core tests."""

from __future__ import annotations

import pytest

from methods.classification.fixed_feature.training import (
    FixedFeatureDataset,
    run_fixed_feature_classification,
)

_CATEGORIES = ["anxiety", "depression", "normal", "suicidal"]
_TRAIN_DATASET = FixedFeatureDataset(
    texts=[
        "anxiety panic worry fear",
        "panic anxiety nervous worry",
        "depression sadness hopeless low",
        "sad depressed hopeless tired",
        "normal calm stable okay",
        "fine normal balanced calm",
        "suicidal self harm die",
        "kill myself suicidal goodbye",
    ],
    labels=[
        "anxiety",
        "anxiety",
        "depression",
        "depression",
        "normal",
        "normal",
        "suicidal",
        "suicidal",
    ],
)


@pytest.mark.parametrize(
    "estimator_config",
    [
        {"name": "logistic_regression", "max_iter": 1000, "class_weight": None},
        {"name": "multinomial_nb", "alpha": 1.0},
        {"name": "decision_tree", "max_depth": None, "min_samples_leaf": 1},
        {"name": "linear_svc", "max_iter": 5000, "class_weight": None},
    ],
)
def test_fixed_feature_classification_trains_four_class_estimators(
    estimator_config: dict[str, object],
) -> None:
    result = run_fixed_feature_classification(
        feature_space_config={
            "name": "tfidf_word",
            "ngram_min": 1,
            "ngram_max": 2,
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
            "lowercase": True,
            "strip_accents": "unicode",
        },
        estimator_config=estimator_config,
        categories=_CATEGORIES,
        train_dataset=_TRAIN_DATASET,
        eval_datasets={"test": _TRAIN_DATASET},
    )

    evaluation = result.evaluations["test"]

    assert evaluation.report["rows_total"] == len(_TRAIN_DATASET.labels)
    assert evaluation.report["accuracy_top_1"] >= 0.75
    assert set(evaluation.predicted_labels).issubset(set(_CATEGORIES))
    assert evaluation.score_matrix.shape == (
        len(_TRAIN_DATASET.labels),
        len(_CATEGORIES),
    )
    assert set(evaluation.report["confusion_matrix"]) == set(_CATEGORIES)
    assert "macro_f1" in evaluation.report


def test_fixed_feature_classification_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="Unknown labels"):
        run_fixed_feature_classification(
            feature_space_config={"name": "tfidf_word", "min_df": 1, "max_df": 1.0},
            estimator_config={"name": "multinomial_nb"},
            categories=_CATEGORIES,
            train_dataset=FixedFeatureDataset(texts=["oops"], labels=["unknown"]),
            eval_datasets={},
        )
