"""중앙 Query SSL labeled/unlabeled split materialization CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.datasets.lib.query_ssl_split import (
    materialize_class_balanced_query_ssl_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a central Query SSL split with class-balanced labeled "
            "rows and a remaining-row unlabeled pool."
        )
    )
    parser.add_argument(
        "--source-train-jsonl",
        required=True,
        type=Path,
        help="Train pool JSONL used as the source for labeled/unlabeled split.",
    )
    parser.add_argument(
        "--source-validation-jsonl",
        required=True,
        type=Path,
        help="Validation JSONL copied into the split artifact.",
    )
    parser.add_argument(
        "--source-test-jsonl",
        required=True,
        type=Path,
        help="Test JSONL copied into the split artifact.",
    )
    parser.add_argument(
        "--split-name",
        required=True,
        help="Stable split directory name.",
    )
    parser.add_argument(
        "--labeled-count-per-class",
        type=int,
        default=1024,
        help="Number of labeled rows to sample from each mapped_label_4 bucket.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic split seed.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/query_ssl_splits"),
        help="Root directory where the split directory is written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = materialize_class_balanced_query_ssl_split(
        source_train_jsonl=args.source_train_jsonl,
        source_validation_jsonl=args.source_validation_jsonl,
        source_test_jsonl=args.source_test_jsonl,
        split_name=args.split_name,
        labeled_count_per_class=args.labeled_count_per_class,
        seed=args.seed,
        output_root=args.output_root,
    )

    print(f"labeled_train_jsonl={artifacts.labeled_train_jsonl}")
    print(f"unlabeled_pool_jsonl={artifacts.unlabeled_pool_jsonl}")
    print(f"validation_jsonl={artifacts.validation_jsonl}")
    print(f"test_jsonl={artifacts.test_jsonl}")
    print(f"manifest_json={artifacts.manifest_json}")
    print(f"summary_json={artifacts.summary_json}")


if __name__ == "__main__":
    main()
