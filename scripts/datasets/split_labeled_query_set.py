"""LabeledQuerySet split CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.datasets.lib.split import build_split_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a labeled query set JSONL into stratified "
            "train/validation JSONL files."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        type=Path,
        help="Path to the labeled query set JSONL.",
    )
    parser.add_argument(
        "--split-name",
        required=True,
        help="Stable split name used for output file names.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation ratio per mapped_label_4 bucket.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets/manual/splits"),
        help="Directory where split outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, validation_path, manifest_path = build_split_artifacts(
        input_jsonl=args.input_jsonl,
        split_name=args.split_name,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"train_jsonl={train_path}")
    print(f"validation_jsonl={validation_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
