"""Wellbeing space-web projection strategy 계약."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from agent.src.contracts.wellbeing_space_web_contracts import (
    WellbeingSpaceWebEdgePayload,
    WellbeingSpaceWebNodePayload,
)
from agent.src.features.inference.interpretation.state import BaselineProfile
from shared.src.domain.entities.inference.events import AnalysisEvent


@dataclass(frozen=True, slots=True)
class SpaceWebProjectionContext:
    """strategy가 같은 canonical 입력을 보도록 묶은 context."""

    recent_events: Sequence[AnalysisEvent]
    baseline_profile: BaselineProfile
    computed_at: datetime
    category_label_overrides: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SpaceWebProjectionResult:
    """strategy가 계산한 graph payload 조각."""

    nodes: tuple[WellbeingSpaceWebNodePayload, ...]
    edges: tuple[WellbeingSpaceWebEdgePayload, ...]


class SpaceWebProjectionStrategy(Protocol):
    """카테고리 score 흐름을 space-web graph로 투영하는 교체 지점."""

    @property
    def name(self) -> str:
        """strategy 식별자."""

    @property
    def version(self) -> str:
        """동일 strategy 안의 계산 버전."""

    def project(self, context: SpaceWebProjectionContext) -> SpaceWebProjectionResult:
        """canonical context를 nodes/edges로 변환한다."""
