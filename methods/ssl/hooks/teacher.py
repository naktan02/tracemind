"""SSL teacher source hook contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


class TeacherPrediction(Protocol):
    """pseudo-label selection이 소비하는 teacher prediction 표면."""

    predicted_label: str
    confidence: float
    margin: float
    runner_up_label: str | None
    runner_up_score: float | None
    raw_scores: dict[str, float]


@dataclass(frozen=True, slots=True)
class TeacherPreparationContext:
    """teacher 준비 hook에 전달하는 실행 문맥."""

    run_id: str
    generated_at: datetime
    seed_rows: tuple[LabeledQueryRow, ...]
    seed_jsonl_ref: str


@dataclass(slots=True)
class PreparedTeacher:
    """teacher source가 준비한 예측 가능 state와 provenance."""

    source_kind: str
    model: Any
    categories: tuple[str, ...]
    outputs: dict[str, str]


class TeacherSource(Protocol):
    """unlabeled row에 pseudo-label evidence를 제공하는 SSL hook."""

    source_kind: str

    def prepare(self, context: TeacherPreparationContext) -> PreparedTeacher:
        """teacher artifact를 재사용하거나 준비한다."""

    def predict_rows(
        self,
        *,
        teacher: PreparedTeacher,
        rows: Sequence[LabeledQueryRow],
    ) -> list[TeacherPrediction]:
        """unlabeled row별 teacher prediction을 반환한다."""
