"""LabeledQuerySet JSONL을 stratified train/validation으로 분리한다."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a labeled query set JSONL into stratified train/validation JSONL files."
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
        default=Path("data/processed/splits"),
        help="Directory where split outputs are written.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def split_rows(
    rows: list[dict[str, Any]],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_label[row["mapped_label_4"]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for label in sorted(rows_by_label):
        bucket = list(rows_by_label[label])
        rng.shuffle(bucket)

        validation_count = int(round(len(bucket) * validation_ratio))
        if validation_count <= 0 and len(bucket) > 1:
            validation_count = 1
        if validation_count >= len(bucket):
            validation_count = len(bucket) - 1

        validation_rows.extend(bucket[:validation_count])
        train_rows.extend(bucket[validation_count:])

    return train_rows, validation_rows


def counts_by_label(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["mapped_label_4"] for row in rows).items()))


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    source_manifest = load_json(args.input_jsonl.with_suffix(".manifest.json"))
    train_rows, validation_rows = split_rows(
        rows,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.split_name}.train.jsonl"
    validation_path = args.output_dir / f"{args.split_name}.validation.jsonl"
    manifest_path = args.output_dir / f"{args.split_name}.manifest.json"

    dump_jsonl(train_path, train_rows)
    dump_jsonl(validation_path, validation_rows)

    manifest = {
        "split_name": args.split_name,
        "source_jsonl": str(args.input_jsonl),
        "source_dataset_id": None if source_manifest is None else source_manifest.get("dataset_id"),
        "source_mapping_version": None
        if source_manifest is None
        else source_manifest.get("mapping_version"),
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "rows_total": len(rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_label_counts": counts_by_label(train_rows),
        "validation_label_counts": counts_by_label(validation_rows),
        "train_jsonl": str(train_path),
        "validation_jsonl": str(validation_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"train_jsonl={train_path}")
    print(f"validation_jsonl={validation_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
