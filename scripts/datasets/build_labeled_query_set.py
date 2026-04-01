"""raw CSV -> labeled query set CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.datasets.lib.label_mapping import build_labeled_query_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw CSV dataset into a mapped LabeledQuerySet JSONL."
    )
    parser.add_argument(
        "--raw-csv",
        required=True,
        type=Path,
        help="Path to the raw source CSV file.",
    )
    parser.add_argument(
        "--mapping-config",
        required=True,
        type=Path,
        help="Path to the TOML mapping config.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/processed/labeled_query_sets"),
        type=Path,
        help="Directory where manifest and JSONL outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path, manifest_path = build_labeled_query_set(
        raw_csv_path=args.raw_csv,
        mapping_config_path=args.mapping_config,
        output_dir=args.output_dir,
    )
    print(f"jsonl={jsonl_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
