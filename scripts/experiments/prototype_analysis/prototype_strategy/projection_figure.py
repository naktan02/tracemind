"""Projection scatter figure writer."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib

from scripts.experiments.prototype_analysis.prototype_strategy.projection_rows import (
    ProjectionOverlayRow,
    ProjectionPointRow,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def draw_projection_figure(
    *,
    point_rows: Sequence[ProjectionPointRow],
    prototype_points: Sequence[ProjectionOverlayRow],
    visual_center_points: Sequence[ProjectionOverlayRow],
    figure_path: Path,
    title: str,
) -> None:
    """Projection points와 optional overlay를 PNG figure로 기록한다."""

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    _draw_labeled_points(point_rows)
    if prototype_points:
        _draw_prototype_points(prototype_points)
    if visual_center_points:
        _draw_visual_center_points(visual_center_points)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def _draw_labeled_points(point_rows: Sequence[ProjectionPointRow]) -> None:
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in point_rows:
        buckets[str(row["label"])].append((float(row["x"]), float(row["y"])))

    for label, points in sorted(buckets.items()):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plt.scatter(xs, ys, s=8, alpha=0.7, label=label)


def _draw_prototype_points(
    prototype_points: Sequence[ProjectionOverlayRow],
) -> None:
    prototype_buckets: dict[str, list[ProjectionOverlayRow]] = defaultdict(list)
    for row in prototype_points:
        prototype_buckets[str(row["label"])].append(row)

    for label, rows in sorted(prototype_buckets.items()):
        xs = [float(row["x"]) for row in rows]
        ys = [float(row["y"]) for row in rows]
        plt.scatter(
            xs,
            ys,
            s=180,
            marker="X",
            edgecolors="black",
            linewidths=0.8,
            label=f"{label} centroid",
        )
        for row in rows:
            plt.annotate(
                _short_prototype_label(str(row["prototype_id"])),
                (float(row["x"]), float(row["y"])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )


def _draw_visual_center_points(
    visual_center_points: Sequence[ProjectionOverlayRow],
) -> None:
    visual_buckets: dict[str, list[ProjectionOverlayRow]] = defaultdict(list)
    for row in visual_center_points:
        visual_buckets[str(row["label"])].append(row)

    for label, rows in sorted(visual_buckets.items()):
        xs = [float(row["x"]) for row in rows]
        ys = [float(row["y"]) for row in rows]
        plt.scatter(
            xs,
            ys,
            s=110,
            marker="D",
            facecolors="none",
            edgecolors="black",
            linewidths=1.1,
            label=f"{label} visual center",
        )
        for row in rows:
            plt.annotate(
                _short_visual_center_label(str(row["visual_center_id"])),
                (float(row["x"]), float(row["y"])),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=8,
            )


def _short_prototype_label(prototype_id: str) -> str:
    parts = prototype_id.split(":")
    if len(parts) >= 3:
        return f"{parts[0]}#{parts[-1]}"
    return parts[0]


def _short_visual_center_label(visual_center_id: str) -> str:
    parts = visual_center_id.split(":")
    if len(parts) >= 3:
        return f"V{parts[0]}#{parts[-1]}"
    return f"V{parts[0]}"
