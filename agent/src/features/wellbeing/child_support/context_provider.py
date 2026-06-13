"""아이용 지원 대화 로컬 컨텍스트 provider."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalRange,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTimeseriesPayload,
)
from agent.src.contracts.wellbeing_space_web_contracts import WellbeingSpaceWebPayload
from agent.src.features.wellbeing.child_support.evidence_summary import (
    ChildSupportEvidenceSummaryBuilder,
)
from agent.src.features.wellbeing.signal.summary_service import WellbeingSummaryService
from agent.src.features.wellbeing.signal.timeseries_service import (
    WellbeingTimeseriesService,
)
from agent.src.features.wellbeing.space_web.projection_service import (
    WellbeingSpaceWebProjectionService,
)
from agent.src.features.wellbeing.storage.child_support_repository import (
    ChildSupportConversationRepository,
    ChildSupportMessageRecord,
)


@dataclass(frozen=True, slots=True)
class ChildSupportConversationContext:
    """child-support 응답 생성에 필요한 agent-local context."""

    conversation_id: str
    wellbeing_summary: WellbeingSignalSummaryPayload | None = None
    wellbeing_summary_is_observed: bool = False
    wellbeing_context_notes: tuple[str, ...] = field(default_factory=tuple)
    recent_messages: tuple[ChildSupportMessageRecord, ...] = field(
        default_factory=tuple
    )


@dataclass(slots=True)
class ChildSupportContextProvider:
    """여러 agent-local source를 child-support context로 조합한다."""

    summary_service: WellbeingSummaryService | None = None
    timeseries_service: WellbeingTimeseriesService | None = None
    space_web_service: WellbeingSpaceWebProjectionService | None = None
    evidence_summary_builder: ChildSupportEvidenceSummaryBuilder | None = None
    conversation_repository: ChildSupportConversationRepository | None = None
    recent_message_limit: int = 8

    def build(self, conversation_id: str) -> ChildSupportConversationContext:
        """현재 conversation에 맞는 로컬 context를 만든다."""

        recent_messages = tuple(self._load_recent_messages(conversation_id))
        summary = self._load_summary()
        return ChildSupportConversationContext(
            conversation_id=conversation_id,
            wellbeing_summary=summary,
            wellbeing_summary_is_observed=self._has_observed_summary(),
            wellbeing_context_notes=self._build_wellbeing_context_notes(summary),
            recent_messages=recent_messages,
        )

    def _load_summary(self) -> WellbeingSignalSummaryPayload | None:
        if self.summary_service is None:
            return None
        try:
            return self.summary_service.get_current_summary()
        except Exception:
            return None

    def _has_observed_summary(self) -> bool:
        if self.summary_service is None:
            return False
        repository = getattr(self.summary_service, "repository", None)
        if repository is not None:
            try:
                return repository.load_latest_summary() is not None
            except Exception:
                return False
        return getattr(self.summary_service, "_mock_payload", None) is not None

    def _load_recent_messages(
        self,
        conversation_id: str,
    ) -> list[ChildSupportMessageRecord]:
        if self.conversation_repository is None:
            return []
        try:
            return self.conversation_repository.get_recent_messages(
                conversation_id,
                limit=self.recent_message_limit,
            )
        except Exception:
            return []

    def _build_wellbeing_context_notes(
        self,
        summary: WellbeingSignalSummaryPayload | None,
    ) -> tuple[str, ...]:
        notes: list[str] = []
        if summary is not None:
            notes.append(
                "현재 전체 신호: "
                f"{summary.signal_level.value}, 점수 {summary.signal_score:.1f}, "
                f"추세 {summary.trend.value}"
            )
        notes.extend(self._build_timeseries_notes())
        notes.extend(self._build_space_web_notes())
        notes.extend(self._build_evidence_notes())
        return tuple(notes)

    def _build_timeseries_notes(self) -> tuple[str, ...]:
        if self.timeseries_service is None:
            return ()
        notes: list[str] = []
        for requested_range in (
            WellbeingSignalRange.LAST_1_DAY,
            WellbeingSignalRange.LAST_7_DAYS,
            WellbeingSignalRange.LAST_14_DAYS,
            WellbeingSignalRange.LAST_30_DAYS,
        ):
            try:
                timeseries = self.timeseries_service.get_timeseries(
                    requested_range=requested_range
                )
            except Exception:
                continue
            note = _timeseries_note(timeseries)
            if note is not None:
                notes.append(note)
        return tuple(notes)

    def _build_space_web_notes(self) -> tuple[str, ...]:
        if self.space_web_service is None:
            return ()
        try:
            space_web = self.space_web_service.get_space_web(
                requested_range=WellbeingSignalRange.LAST_7_DAYS
            )
        except Exception:
            return ()
        return _space_web_notes(space_web)

    def _build_evidence_notes(self) -> tuple[str, ...]:
        if self.evidence_summary_builder is None:
            return ()
        try:
            evidence_summary = self.evidence_summary_builder.build()
        except Exception:
            return ()
        return evidence_summary.to_prompt_lines()


def _timeseries_note(timeseries: WellbeingSignalTimeseriesPayload) -> str | None:
    points = timeseries.points
    if len(points) < 2:
        return None
    first = points[0].signal_score
    last = points[-1].signal_score
    delta = last - first
    max_score = max(point.signal_score for point in points)
    min_score = min(point.signal_score for point in points)
    return (
        f"{timeseries.range.value} 변화: 시작 {first:.1f}, 현재 {last:.1f}, "
        f"변화 {delta:+.1f}, 범위 {min_score:.1f}~{max_score:.1f}"
    )


def _space_web_notes(space_web: WellbeingSpaceWebPayload) -> tuple[str, ...]:
    notes: list[str] = []
    top_nodes = sorted(
        space_web.nodes,
        key=lambda node: node.intensity,
        reverse=True,
    )[:4]
    if top_nodes:
        notes.append(
            "최근 7일 주요 축: "
            + ", ".join(
                f"{node.label} {node.intensity:.1f}/{node.trend.value}"
                for node in top_nodes
            )
        )
    top_edges = sorted(
        space_web.edges,
        key=lambda edge: edge.weight,
        reverse=True,
    )[:3]
    if top_edges:
        notes.append(
            "최근 7일 같이 움직인 축: "
            + ", ".join(
                f"{edge.source}-{edge.target} {edge.weight:.1f}" for edge in top_edges
            )
        )
    return tuple(notes)
