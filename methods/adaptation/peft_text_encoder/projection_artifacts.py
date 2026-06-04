"""PEFT text encoder/head final representation projection artifact writer."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from methods.adaptation.text_encoder_classifier.projection import (
    collect_pooled_classifier_features,
    reduce_features_2d,
)


def write_peft_encoder_projection_artifacts(
    *,
    model: Any,
    eval_loaders: Mapping[str, Any] | None,
    categories: list[str],
    device: str,
    projection_dir: Path,
    seed: int,
    schema_version: str,
) -> dict[str, Any] | None:
    """최종 PEFT encoder representation을 2D projection artifact로 저장한다."""

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
            title=(
                f"{dataset_name} final PEFT encoder representation "
                f"({projection['reducer']})"
            ),
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
        "projection_space": "final_peft_encoder_pooled_backbone_features",
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
    collected = collect_pooled_classifier_features(
        model=model,
        dataloader=dataloader,
        categories=categories,
        device=device,
    )
    projection = reduce_features_2d(
        features=collected.features,
        reducer_name="umap",
        seed=seed,
        n_neighbors=15,
    )
    rows = []
    for index, (x, y) in enumerate(projection.coordinates):
        true_label = collected.labels[index]
        predicted_label = collected.predicted_labels[index]
        rows.append(
            {
                "row_index": index,
                "x": round(float(x), 6),
                "y": round(float(y), 6),
                "label": true_label,
                "predicted_label": predicted_label,
                "is_correct": true_label == predicted_label,
                "top_1_probability": round(
                    float(collected.top_1_probabilities[index]),
                    6,
                ),
            }
        )
    return {
        "rows": rows,
        "reducer": projection.reducer,
        "fallback_reason": projection.fallback_reason,
    }


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
