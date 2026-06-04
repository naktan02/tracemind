from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

BALANCED_TEST_JSONL = Path(
    "data/datasets/ourafla_mental_health/query_ssl/"
    "labeled1024_per_class_seed42_v1/"
    "test_balanced_validation_test_seed42.jsonl"
)
BALANCED_TEST_SUMMARY_JSON = Path(
    "data/datasets/ourafla_mental_health/query_ssl/"
    "labeled1024_per_class_seed42_v1/"
    "test_balanced_validation_test_seed42.summary.json"
)


def test_balanced_validation_test_eval_set_has_equal_class_counts() -> None:
    if not BALANCED_TEST_JSONL.exists() or not BALANCED_TEST_SUMMARY_JSON.exists():
        pytest.skip(
            "balanced eval dataset is gitignored and only verified when present"
        )

    summary = json.loads(BALANCED_TEST_SUMMARY_JSON.read_text(encoding="utf-8"))
    expected_counts = {
        "anxiety": 798,
        "depression": 798,
        "normal": 798,
        "suicidal": 798,
    }

    counts: Counter[str] = Counter()
    with BALANCED_TEST_JSONL.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                counts[json.loads(line)["mapped_label_4"]] += 1

    assert dict(sorted(counts.items())) == expected_counts
    assert summary["balanced_label_counts"] == expected_counts
    assert summary["row_count"] == 3192
    assert summary["selection_policy"]["per_class_count"] == 798
