"""labeled query row JSONL canonical contract와 IO helper."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import NotRequired, TypedDict


class LabeledQueryRow(TypedDict):
    """실험/적응/프로토타입 레일이 공유하는 labeled query row shape."""

    query_id: str
    text: str
    raw_label_scheme: str
    raw_label: str
    mapped_label_4: str
    locale: str
    annotation_source: str
    approved_by: str | None
    created_at: str
    aug_0: NotRequired[str]
    aug_1: NotRequired[str]
    aug_0_pivot_lang: NotRequired[str]
    aug_1_pivot_lang: NotRequired[str]
    weak_text: NotRequired[str]
    strong_text: NotRequired[str]
    weak_translated_text: NotRequired[str]
    strong_translated_text: NotRequired[str]


def load_labeled_query_rows(path: str | Path) -> list[LabeledQueryRow]:
    """JSONL 파일을 labeled query row 목록으로 읽는다."""

    resolved_path = Path(str(path))
    rows: list[LabeledQueryRow] = []
    for line in resolved_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def dump_labeled_query_rows(
    path: str | Path,
    rows: Sequence[LabeledQueryRow],
) -> None:
    """labeled query row 목록을 JSONL로 기록한다."""

    resolved_path = Path(str(path))
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def get_labeled_query_row_mapped_label(row: LabeledQueryRow) -> str:
    """canonical 4-class label 값을 반환한다."""

    return str(row["mapped_label_4"])


def group_labeled_query_rows_by_label(
    rows: Sequence[LabeledQueryRow],
) -> dict[str, list[LabeledQueryRow]]:
    """mapped_label_4 기준으로 row를 묶는다."""

    rows_by_label: dict[str, list[LabeledQueryRow]] = defaultdict(list)
    for row in rows:
        rows_by_label[get_labeled_query_row_mapped_label(row)].append(row)
    return dict(sorted(rows_by_label.items()))


def count_labeled_query_rows_by_label(
    rows: Sequence[LabeledQueryRow],
) -> dict[str, int]:
    """mapped_label_4 기준 row 개수를 센다."""

    return dict(
        sorted(Counter(get_labeled_query_row_mapped_label(row) for row in rows).items())
    )
