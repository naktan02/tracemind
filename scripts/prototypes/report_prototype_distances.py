"""PrototypePack의 카테고리 간 pairwise 거리/유사도 표를 출력한다."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# 스크립트 직접 실행 시 프로젝트 루트를 import path에 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.src.contracts.prototype_contracts import (  # noqa: E402
    extract_category_centroids,
    load_prototype_pack_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report pairwise cosine similarity and L2 distance for a prototype "
            "pack."
        )
    )
    parser.add_argument(
        "--prototype-pack",
        required=True,
        type=Path,
        help="Path to the prototype pack JSON file.",
    )
    return parser.parse_args()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot_product / (left_norm * right_norm)


def l2_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left, right, strict=True)
        )
    )


def render_table(
    *,
    title: str,
    categories: list[str],
    values: dict[tuple[str, str], float],
) -> str:
    header = ["category"] + categories
    lines = [
        title,
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row_category in categories:
        row = [row_category]
        for column_category in categories:
            row.append(f"{values[(row_category, column_category)]:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = load_prototype_pack_payload(args.prototype_pack)
    centroids = extract_category_centroids(payload)
    categories = sorted(centroids)

    cosine_values: dict[tuple[str, str], float] = {}
    l2_values: dict[tuple[str, str], float] = {}
    for row_category in categories:
        for column_category in categories:
            left = centroids[row_category]
            right = centroids[column_category]
            cosine_values[(row_category, column_category)] = cosine_similarity(
                left,
                right,
            )
            l2_values[(row_category, column_category)] = l2_distance(left, right)

    print(
        render_table(
            title="cosine_similarity",
            categories=categories,
            values=cosine_values,
        )
    )
    print()
    print(
        render_table(
            title="l2_distance",
            categories=categories,
            values=l2_values,
        )
    )


if __name__ == "__main__":
    main()
