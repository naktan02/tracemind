"""dataset 다운로드 재사용 함수."""

from __future__ import annotations

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
