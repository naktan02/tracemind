"""Text encoder classifier representation projection primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA


@dataclass(frozen=True, slots=True)
class PooledClassifierFeatures:
    """Pooled representation과 classifier 예측 결과."""

    features: np.ndarray
    labels: tuple[str, ...]
    predicted_labels: tuple[str, ...]
    top_1_probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FeatureProjection2D:
    """2D reducer 결과와 fallback 정보."""

    coordinates: np.ndarray
    reducer: str
    fallback_reason: str | None


def collect_pooled_classifier_features(
    *,
    model: Any,
    dataloader: Any,
    categories: list[str],
    device: str,
    collect_labels: bool = True,
) -> PooledClassifierFeatures:
    """text encoder pooled feature와 top-1 prediction을 수집한다."""

    model.eval()
    features: list[np.ndarray] = []
    labels: list[str] = []
    predicted_labels: list[str] = []
    top_1_probabilities: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pooled = model.extract_pooled_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = model.classifier(pooled)
            probabilities = torch.softmax(logits, dim=-1)
            top_values, top_indices = torch.topk(probabilities, k=1, dim=-1)
            features.append(pooled.detach().float().cpu().numpy())
            if collect_labels:
                label_indices = batch["labels"].to(device)
                labels.extend(
                    categories[index] for index in label_indices.cpu().tolist()
                )
            predicted_labels.extend(
                categories[index] for index in top_indices[:, 0].cpu().tolist()
            )
            top_1_probabilities.extend(top_values[:, 0].cpu().tolist())

    if not features:
        feature_array = np.zeros((0, 0), dtype=np.float32)
    else:
        feature_array = np.concatenate(features, axis=0)
    return PooledClassifierFeatures(
        features=feature_array,
        labels=tuple(labels),
        predicted_labels=tuple(predicted_labels),
        top_1_probabilities=tuple(top_1_probabilities),
    )


def reduce_features_2d(
    *,
    features: np.ndarray,
    reducer_name: str = "umap",
    seed: int,
    n_neighbors: int = 15,
) -> FeatureProjection2D:
    """feature matrix를 2D 좌표로 줄인다."""

    row_count = int(features.shape[0])
    if row_count == 0:
        return FeatureProjection2D(
            coordinates=np.zeros((0, 2), dtype=np.float32),
            reducer="none",
            fallback_reason="empty_features",
        )
    if row_count < 3:
        return FeatureProjection2D(
            coordinates=zero_pad_projection(features),
            reducer="identity_zero_pad",
            fallback_reason="row_count_lt_3",
        )
    if reducer_name == "pca":
        return FeatureProjection2D(
            coordinates=pca_projection(features, seed=seed),
            reducer="pca",
            fallback_reason=None,
        )
    if reducer_name != "umap":
        raise ValueError(f"Unsupported reducer: {reducer_name!r}")
    try:
        from umap import UMAP

        reducer = UMAP(
            n_components=2,
            n_neighbors=max(2, min(int(n_neighbors), row_count - 1)),
            random_state=seed,
        )
        return FeatureProjection2D(
            coordinates=reducer.fit_transform(features),
            reducer="umap",
            fallback_reason=None,
        )
    except Exception as exc:  # pragma: no cover - optional stack/runtime fallback
        return FeatureProjection2D(
            coordinates=pca_projection(features, seed=seed),
            reducer="pca",
            fallback_reason=f"umap_failed:{exc}",
        )


def pca_projection(features: np.ndarray, *, seed: int) -> np.ndarray:
    """PCA 2D projection을 계산한다."""

    row_count = int(features.shape[0])
    feature_count = int(features.shape[1]) if features.ndim == 2 else 0
    component_count = min(2, row_count, feature_count)
    if component_count < 2:
        return zero_pad_projection(features)
    return PCA(n_components=2, random_state=seed).fit_transform(features)


def zero_pad_projection(features: np.ndarray) -> np.ndarray:
    """2D reducer를 적용할 수 없는 작은 feature를 2D로 zero-pad한다."""

    row_count = int(features.shape[0])
    coordinates = np.zeros((row_count, 2), dtype=np.float32)
    if features.ndim == 2 and features.shape[1] > 0:
        coordinates[:, 0] = features[:, 0]
    return coordinates
