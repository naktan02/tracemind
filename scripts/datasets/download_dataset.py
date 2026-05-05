"""HuggingFace 데이터셋 다운로드 CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.datasets.lib.download import download_huggingface_dataset_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset to local CSV."
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help=(
            "HuggingFace dataset identifier "
            "(e.g. ourafla/Mental-Health_Text-Classification_Dataset)."
        ),
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional file name inside the dataset repo to map onto the given split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where CSV output is written.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit CSV output path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("hf_cache"),
        help="HuggingFace cache directory.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HuggingFace dataset revision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_huggingface_dataset_to_csv(
        dataset_id=args.dataset_id,
        split=args.split,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        data_file=args.data_file,
        output_path=args.output_path,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
