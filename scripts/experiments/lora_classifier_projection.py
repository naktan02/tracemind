"""LoRA-classifier final representation projection artifact writer."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_lora_classifier_projection_artifacts(
    *,
    model: Any,
    eval_loaders: Mapping[str, Any] | None,
    categories: list[str],
    device: str,
    projection_dir: Path,
    seed: int,
    schema_version: str,
) -> dict[str, Any] | None:
    """최종 LoRA representation을 eval set별 2D projection artifact로 저장한다."""

    if eval_loaders is None:
        return None
    if not hasattr(model, "extract_pooled_features"):
        return {
            "enabled": False,
            "reason": "model_does_not_expose_extract_pooled_features",
        }

    projection_entries: dict[str, dict[str, Any]] = {}
    for dataset_name, dataloader in eval_loaders.items():
        projection = _build_dataset_projection(
            model=model,
            dataloader=dataloader,
            categories=categories,
            device=device,
            seed=seed,
        )
        safe_name = _safe_filename(dataset_name)
        points_path = projection_dir / f"{safe_name}.projection.jsonl"
        figure_path = projection_dir / f"{safe_name}.projection.png"
        _write_projection_points(points_path, projection["rows"])
        _draw_projection_figure(
            figure_path=figure_path,
            rows=projection["rows"],
            title=f"{dataset_name} final LoRA representation ({projection['reducer']})",
        )
        projection_entries[dataset_name] = {
            "reducer": projection["reducer"],
            "fallback_reason": projection["fallback_reason"],
            "row_count": len(projection["rows"]),
            "points_jsonl": str(points_path),
            "figure_png": str(figure_path),
        }

    manifest = {
        "schema_version": schema_version,
        "projection_space": "final_lora_pooled_backbone_features",
        "datasets": projection_entries,
    }
    manifest_path = projection_dir / "projection_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {
        "enabled": True,
        "manifest_path": str(manifest_path),
        "datasets": projection_entries,
    }


def _build_dataset_projection(
    *,
    model: Any,
    dataloader: Any,
    categories: list[str],
    device: str,
    seed: int,
) -> dict[str, Any]:
    features, labels, predicted_labels, confidences = _collect_features(
        model=model,
        dataloader=dataloader,
        categories=categories,
        device=device,
    )
    coordinates, reducer, fallback_reason = _reduce_features(
        features=features,
        seed=seed,
    )
    rows = []
    for index, (x, y) in enumerate(coordinates):
        true_label = labels[index]
        predicted_label = predicted_labels[index]
        rows.append(
            {
                "row_index": index,
                "x": round(float(x), 6),
                "y": round(float(y), 6),
                "label": true_label,
                "predicted_label": predicted_label,
                "is_correct": true_label == predicted_label,
                "top_1_probability": round(float(confidences[index]), 6),
            }
        )
    return {
        "rows": rows,
        "reducer": reducer,
        "fallback_reason": fallback_reason,
    }


def _collect_features(
    *,
    model: Any,
    dataloader: Any,
    categories: list[str],
    device: str,
) -> tuple[np.ndarray, list[str], list[str], list[float]]:
    model.eval()
    features: list[np.ndarray] = []
    labels: list[str] = []
    predicted_labels: list[str] = []
    confidences: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_indices = batch["labels"].to(device)
            pooled = model.extract_pooled_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = model.classifier(pooled)
            probabilities = torch.softmax(logits, dim=-1)
            top_values, top_indices = torch.topk(probabilities, k=1, dim=-1)
            features.append(pooled.detach().float().cpu().numpy())
            labels.extend(categories[index] for index in label_indices.cpu().tolist())
            predicted_labels.extend(
                categories[index] for index in top_indices[:, 0].cpu().tolist()
            )
            confidences.extend(top_values[:, 0].cpu().tolist())

    if not features:
        return np.zeros((0, 2), dtype=np.float32), [], [], []
    return np.concatenate(features, axis=0), labels, predicted_labels, confidences


def _reduce_features(
    *,
    features: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, str, str | None]:
    row_count = int(features.shape[0])
    if row_count == 0:
        return np.zeros((0, 2), dtype=np.float32), "none", "empty_dataset"
    if row_count < 3:
        return _zero_pad_projection(features), "identity_zero_pad", "row_count_lt_3"

    try:
        from umap import UMAP

        reducer = UMAP(
            n_components=2,
            n_neighbors=max(2, min(15, row_count - 1)),
            random_state=seed,
        )
        return reducer.fit_transform(features), "umap", None
    except Exception as exc:  # pragma: no cover - fallback depends on optional stack
        return _pca_projection(features, seed=seed), "pca", f"umap_failed:{exc}"


def _pca_projection(features: np.ndarray, *, seed: int) -> np.ndarray:
    row_count = int(features.shape[0])
    feature_count = int(features.shape[1]) if features.ndim == 2 else 0
    component_count = min(2, row_count, feature_count)
    if component_count < 2:
        return _zero_pad_projection(features)
    return PCA(n_components=2, random_state=seed).fit_transform(features)


def _zero_pad_projection(features: np.ndarray) -> np.ndarray:
    row_count = int(features.shape[0])
    coordinates = np.zeros((row_count, 2), dtype=np.float32)
    if features.ndim == 2 and features.shape[1] > 0:
        coordinates[:, 0] = features[:, 0]
    return coordinates


def _write_projection_points(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def _draw_projection_figure(
    *,
    figure_path: Path,
    rows: list[dict[str, Any]],
    title: str,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["label"])].append(row)
    for label, label_rows in sorted(buckets.items()):
        plt.scatter(
            [float(row["x"]) for row in label_rows],
            [float(row["y"]) for row in label_rows],
            s=10,
            alpha=0.72,
            label=label,
        )

    incorrect_rows = [row for row in rows if not bool(row["is_correct"])]
    if incorrect_rows:
        plt.scatter(
            [float(row["x"]) for row in incorrect_rows],
            [float(row["y"]) for row in incorrect_rows],
            s=28,
            facecolors="none",
            edgecolors="black",
            linewidths=0.7,
            label="incorrect",
        )

    plt.title(title)
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def _safe_filename(value: str) -> str:
    characters = [
        character.lower() if character.isalnum() else "_" for character in value.strip()
    ]
    return "".join(characters).strip("_") or "dataset"
