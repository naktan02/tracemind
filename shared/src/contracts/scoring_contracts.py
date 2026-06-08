"""Scoring backend/policy payload 계약."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field


class ScoringConfigPayload(BaseModel):
    """분류 score 계산 runtime 설정.

    학습 objective가 아니라 inference/validation에서 category score를 어떤
    backend와 정책으로 계산할지 표현한다.
    """

    model_config = ConfigDict(extra="forbid")

    scorer_backend_name: str = Field(description="Category score 계산 backend.")
    score_policy_name: str | None = Field(
        default=None,
        description="Backend score를 집계/변환하는 정책 이름.",
    )
    score_top_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-k score 정책이 사용할 k 값.",
    )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "ScoringConfigPayload":
        """Mapping 입력을 scoring config로 정규화한다."""

        if source is None:
            raise ValueError("scorer_backend_name is required.")
        backend_name = _required_str(source.get("scorer_backend_name"))
        return cls(
            scorer_backend_name=backend_name,
            score_policy_name=_optional_str(source.get("score_policy_name")),
            score_top_k=_optional_positive_int(source.get("score_top_k")),
        )

    def to_mapping(self) -> dict[str, str | int]:
        """저장/전송용 flat mapping으로 변환한다."""

        result: dict[str, str | int] = {"scorer_backend_name": self.scorer_backend_name}
        if self.score_policy_name is not None:
            result["score_policy_name"] = self.score_policy_name
        if self.score_top_k is not None:
            result["score_top_k"] = self.score_top_k
        return result


def _required_str(value: object) -> str:
    normalized = "" if value is None else str(value).strip()
    if not normalized:
        raise ValueError("scorer_backend_name is required.")
    return normalized


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("score_top_k must not be bool.")
    normalized = int(value)
    if normalized < 1:
        raise ValueError("score_top_k must be >= 1.")
    return normalized
