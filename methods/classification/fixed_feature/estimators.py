"""Fixed-feature 분류용 scikit-learn estimator 생성."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def build_estimator(config: Mapping[str, Any]) -> Any:
    """Hydra leaf mapping에서 classifier estimator를 생성한다."""

    name = str(config.get("name", ""))
    if name == "logistic_regression":
        return LogisticRegression(
            C=float(config.get("c", 1.0)),
            max_iter=int(config.get("max_iter", 1000)),
            class_weight=config.get("class_weight", None),
            solver=str(config.get("solver", "lbfgs")),
            random_state=int(config.get("random_state", 42)),
        )
    if name == "multinomial_nb":
        return MultinomialNB(alpha=float(config.get("alpha", 1.0)))
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=config.get("max_depth", None),
            min_samples_leaf=int(config.get("min_samples_leaf", 1)),
            class_weight=config.get("class_weight", None),
            random_state=int(config.get("random_state", 42)),
        )
    if name == "linear_svc":
        return LinearSVC(
            C=float(config.get("c", 1.0)),
            max_iter=int(config.get("max_iter", 5000)),
            class_weight=config.get("class_weight", None),
            random_state=int(config.get("random_state", 42)),
        )
    raise ValueError(f"Unsupported fixed-feature estimator: {name}")
