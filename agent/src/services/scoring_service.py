"""프로토타입 점수 계산 서비스 자리표시자."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class ScoringService:
    """임베딩과 카테고리 프로토타입 간 점수를 계산한다."""

    similarity_name: str = "cosine"

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float]],
    ) -> dict[str, float]:
        raise NotImplementedError("Prototype similarity scoring is not implemented yet.")
