"""Baseline 대비 동시 상승으로 space-web edge를 계산하는 기본 strategy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations

from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalLevel,
    WellbeingSignalTrend,
)
from agent.src.contracts.wellbeing_space_web_contracts import (
    WellbeingSpaceWebEdgePayload,
    WellbeingSpaceWebNodePayload,
    WellbeingSpaceWebRelationType,
)
from agent.src.services.wellbeing.space_web.projection_strategy import (
    SpaceWebProjectionContext,
    SpaceWebProjectionResult,
)


@dataclass(frozen=True, slots=True)
class CoactivationDeltaConfig:
    """coactivation-delta strategy의 해석 정책."""

    ignored_categories: frozenset[str] = field(
        default_factory=lambda: frozenset({"normal"})
    )
    positive_delta_floor: float = 0.02
    trend_delta_threshold: float = 0.08
    intensity_delta_weight: float = 0.3
    edge_normalization_scale: float = 0.16
    min_edge_weight: float = 8.0
    min_edge_evidence_count: int = 1


@dataclass(slots=True)
class CoactivationDeltaSpaceWebStrategy:
    """같이 올라간 카테고리 쌍을 edge로 해석하는 v1 strategy."""

    config: CoactivationDeltaConfig = field(default_factory=CoactivationDeltaConfig)
    category_label_overrides: Mapping[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "coactivation_delta"

    @property
    def version(self) -> str:
        return "v1"

    def project(self, context: SpaceWebProjectionContext) -> SpaceWebProjectionResult:
        categories = _collect_categories(
            context=context,
            ignored_categories=self.config.ignored_categories,
        )
        if not categories:
            return SpaceWebProjectionResult(nodes=(), edges=())

        nodes = tuple(
            self._build_node(
                category=category,
                context=context,
            )
            for category in categories
        )
        edges = tuple(
            self._build_edge(source=source, target=target, context=context)
            for source, target in combinations(categories, 2)
        )
        visible_edges = tuple(edge for edge in edges if edge is not None)
        return SpaceWebProjectionResult(nodes=nodes, edges=visible_edges)

    def _build_node(
        self,
        *,
        category: str,
        context: SpaceWebProjectionContext,
    ) -> WellbeingSpaceWebNodePayload:
        scores = [
            float(event.category_scores[category])
            for event in context.recent_events
            if category in event.category_scores
        ]
        recent_average = _average(scores)
        baseline = context.baseline_profile.category_means.get(category, recent_average)
        positive_delta = max(recent_average - baseline, 0.0)
        intensity = _clamp_0_100(
            (
                (1.0 - self.config.intensity_delta_weight) * recent_average
                + self.config.intensity_delta_weight * positive_delta
            )
            * 100.0
        )

        return WellbeingSpaceWebNodePayload(
            id=category,
            label=_resolve_category_label(
                category=category,
                context_overrides=context.category_label_overrides,
                strategy_overrides=self.category_label_overrides,
            ),
            intensity=round(intensity, 2),
            level=_level_from_intensity(intensity),
            trend=_trend_from_scores(
                scores=scores,
                threshold=self.config.trend_delta_threshold,
            ),
            observed_event_count=len(scores),
        )

    def _build_edge(
        self,
        *,
        source: str,
        target: str,
        context: SpaceWebProjectionContext,
    ) -> WellbeingSpaceWebEdgePayload | None:
        coactivation_total = 0.0
        evidence_count = 0

        for event in context.recent_events:
            source_delta = _positive_event_delta(
                category=source,
                scores=event.category_scores,
                baseline_means=context.baseline_profile.category_means,
                floor=self.config.positive_delta_floor,
            )
            target_delta = _positive_event_delta(
                category=target,
                scores=event.category_scores,
                baseline_means=context.baseline_profile.category_means,
                floor=self.config.positive_delta_floor,
            )
            if source_delta <= 0.0 or target_delta <= 0.0:
                continue
            evidence_count += 1
            coactivation_total += source_delta * target_delta

        if evidence_count < self.config.min_edge_evidence_count:
            return None

        average_strength = coactivation_total / max(len(context.recent_events), 1)
        weight = _clamp_0_100(
            average_strength / self.config.edge_normalization_scale * 100.0
        )
        if weight < self.config.min_edge_weight:
            return None

        return WellbeingSpaceWebEdgePayload(
            source=source,
            target=target,
            weight=round(weight, 2),
            relation_type=WellbeingSpaceWebRelationType.COACTIVATION,
            evidence_count=evidence_count,
        )


def _collect_categories(
    *,
    context: SpaceWebProjectionContext,
    ignored_categories: frozenset[str],
) -> tuple[str, ...]:
    categories: set[str] = set(context.baseline_profile.category_means)
    for event in context.recent_events:
        categories.update(event.category_scores)
    return tuple(
        sorted(
            category for category in categories if category not in ignored_categories
        )
    )


def _resolve_category_label(
    *,
    category: str,
    context_overrides: Mapping[str, str],
    strategy_overrides: Mapping[str, str],
) -> str:
    return (
        context_overrides.get(category)
        or strategy_overrides.get(category)
        or category.replace("_", " ")
    )


def _positive_event_delta(
    *,
    category: str,
    scores: Mapping[str, float],
    baseline_means: Mapping[str, float],
    floor: float,
) -> float:
    if category not in scores:
        return 0.0
    current_score = float(scores[category])
    delta = current_score - baseline_means.get(category, current_score)
    return delta if delta >= floor else 0.0


def _trend_from_scores(
    *,
    scores: list[float],
    threshold: float,
) -> WellbeingSignalTrend:
    if len(scores) < 2:
        return WellbeingSignalTrend.UNKNOWN
    midpoint = max(len(scores) // 2, 1)
    first_average = _average(scores[:midpoint])
    second_average = _average(scores[midpoint:])
    delta = second_average - first_average
    if delta >= threshold:
        return WellbeingSignalTrend.RISING
    if delta <= -threshold:
        return WellbeingSignalTrend.FALLING
    return WellbeingSignalTrend.STEADY


def _level_from_intensity(intensity: float) -> WellbeingSignalLevel:
    if intensity >= 80.0:
        return WellbeingSignalLevel.VERY_HIGH
    if intensity >= 55.0:
        return WellbeingSignalLevel.HIGH
    if intensity >= 35.0:
        return WellbeingSignalLevel.MODERATE
    return WellbeingSignalLevel.LOW


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _clamp_0_100(value: float) -> float:
    return min(max(value, 0.0), 100.0)
