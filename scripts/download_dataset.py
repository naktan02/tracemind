"""HuggingFace 데이터셋을 로컬 CSV로 다운로드한다."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset to local CSV."
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="HuggingFace dataset identifier (e.g. ourafla/Mental-Health_Text-Classification_Dataset).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where CSV output is written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("hf_cache"),
        help="HuggingFace cache directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    print(f"다운로드 시작: {args.dataset_id} (split={args.split})", flush=True)
    ds = load_dataset(
        args.dataset_id,
        split=args.split,
        cache_dir=str(args.cache_dir),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # dataset_id에서 파일명 생성: 슬래시 → 언더스코어
    safe_name = args.dataset_id.replace("/", "_")
    output_path = args.output_dir / f"{safe_name}.csv"

    ds.to_csv(str(output_path), index=False)
    print(f"저장 완료: {output_path} ({len(ds)} rows)", flush=True)


if __name__ == "__main__":
    main()
