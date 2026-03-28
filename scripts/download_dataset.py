"""HuggingFace 데이터셋을 로컬 CSV로 다운로드한다."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_default_output_path(
    *,
    dataset_id: str,
    split: str,
    output_dir: Path,
) -> Path:
    """dataset_id와 split을 포함한 기본 출력 경로를 만든다."""
    safe_name = dataset_id.replace("/", "_")
    return output_dir / f"{safe_name}__{split}.csv"


def download_huggingface_dataset_to_csv(
    *,
    dataset_id: str,
    split: str,
    output_dir: Path,
    cache_dir: Path,
    data_file: str | None = None,
    output_path: Path | None = None,
    revision: str | None = None,
) -> Path:
    """HuggingFace 데이터셋 split 하나를 CSV로 저장한다."""
    from datasets import load_dataset

    resolved_output_path = output_path or build_default_output_path(
        dataset_id=dataset_id,
        split=split,
        output_dir=output_dir,
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, object] = {
        "path": dataset_id,
        "split": split,
        "cache_dir": str(cache_dir),
    }
    if data_file is not None:
        load_kwargs["data_files"] = {split: data_file}
    if revision is not None:
        load_kwargs["revision"] = revision

    print(
        f"다운로드 시작: dataset={dataset_id} split={split} "
        f"data_file={data_file or '-'}",
        flush=True,
    )
    ds = load_dataset(**load_kwargs)
    ds.to_csv(str(resolved_output_path), index=False)
    print(f"저장 완료: {resolved_output_path} ({len(ds)} rows)", flush=True)
    return resolved_output_path


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
