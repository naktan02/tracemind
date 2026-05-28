"""PrototypePack의 카테고리 간 pairwise 거리/유사도 표를 출력한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from methods.prototype.distance_report import (  # noqa: E402
    SUPPORTED_PROTOTYPE_CENTROID_VIEWS,
    build_pairwise_distance_report,
    render_pairwise_table,
    resolve_prototype_centroid_view,
)
from shared.src.contracts.prototype_contracts import (  # noqa: E402
    load_prototype_pack_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report pairwise cosine similarity and L2 distance for a prototype pack."
        )
    )
    parser.add_argument(
        "--prototype-pack",
        required=True,
        type=Path,
        help="Path to the prototype pack JSON file.",
    )
    parser.add_argument(
        "--centroid-view",
        choices=SUPPORTED_PROTOTYPE_CENTROID_VIEWS,
        default="strict_single",
        help="How to derive one centroid per category from the canonical pack.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_prototype_pack_payload(args.prototype_pack)
    centroids = resolve_prototype_centroid_view(
        payload=payload,
        centroid_view=args.centroid_view,
    )
    report = build_pairwise_distance_report(centroids)

    print(
        render_pairwise_table(
            title="cosine_similarity",
            categories=report.categories,
            values=report.cosine_values,
        )
    )
    print()
    print(
        render_pairwise_table(
            title="l2_distance",
            categories=report.categories,
            values=report.l2_values,
        )
    )


if __name__ == "__main__":
    main()
