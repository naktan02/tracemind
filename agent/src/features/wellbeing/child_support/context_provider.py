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

_MAX_COUNSELOR_LINE_CHARS = 180


@dataclass(frozen=True, slots=True)
class ChildSupportCounselorContext:
    """LLM이 상담사처럼 읽을 수 있게 재구성한 현재 이해."""

    known_child_statements: tuple[str, ...] = field(default_factory=tuple)
    observed_patterns: tuple[str, ...] = field(default_factory=tuple)
    inferred_state: tuple[str, ...] = field(default_factory=tuple)
    open_questions: tuple[str, ...] = field(default_factory=tuple)
    response_cautions: tuple[str, ...] = field(default_factory=tuple)

    def to_prompt_lines(self) -> tuple[str, ...]:
        lines = ["상담사의 현재 이해:"]
        lines.extend(
            _section_lines("아이가 직접 말한 내용", self.known_child_statements)
        )
        lines.extend(_section_lines("최근 흐름으로 보이는 것", self.observed_patterns))
        lines.extend(_section_lines("현재 상태로 추정되는 것", self.inferred_state))
        lines.extend(_section_lines("아직 확인되지 않은 것", self.open_questions))
        lines.extend(_section_lines("응답할 때 주의할 점", self.response_cautions))
        return tuple(lines)


@dataclass(frozen=True, slots=True)
class ChildSupportConversationContext:
    """child-support 응답 생성에 필요한 agent-local context."""

    conversation_id: str
    wellbeing_summary: WellbeingSignalSummaryPayload | None = None
    wellbeing_summary_is_observed: bool = False
    wellbeing_context_notes: tuple[str, ...] = field(default_factory=tuple)
    counselor_context: ChildSupportCounselorContext = field(
        default_factory=ChildSupportCounselorContext
    )
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
        wellbeing_context_notes = self._build_wellbeing_context_notes(summary)
        return ChildSupportConversationContext(
            conversation_id=conversation_id,
            wellbeing_summary=summary,
            wellbeing_summary_is_observed=self._has_observed_summary(),
            wellbeing_context_notes=wellbeing_context_notes,
            counselor_context=_build_counselor_context(
                summary=summary,
                notes=wellbeing_context_notes,
                recent_messages=recent_messages,
            ),
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


def _build_counselor_context(
    *,
    summary: WellbeingSignalSummaryPayload | None,
    notes: tuple[str, ...],
    recent_messages: tuple[ChildSupportMessageRecord, ...],
) -> ChildSupportCounselorContext:
    child_statements = tuple(
        _shorten_context_line(message.text)
        for message in recent_messages
        if message.role == "child" and message.text.strip()
    )[-4:]
    observed_patterns = _build_observed_patterns(summary=summary, notes=notes)
    inferred_state = _build_inferred_state(
        notes=notes, child_statements=child_statements
    )
    open_questions = _build_open_questions(notes=notes)
    return ChildSupportCounselorContext(
        known_child_statements=child_statements,
        observed_patterns=observed_patterns,
        inferred_state=inferred_state,
        open_questions=open_questions,
        response_cautions=(
            "이미 아이가 말한 내용을 모르는 것처럼 다시 묻기 전에 "
            "알고 있는 범위를 먼저 말함",
            "원문이나 점수를 길게 나열하지 않고 아이가 느낄 현재 상태로 번역해서 말함",
            "확인되지 않은 사실은 단정하지 않음",
        ),
    )


def _build_observed_patterns(
    *,
    summary: WellbeingSignalSummaryPayload | None,
    notes: tuple[str, ...],
) -> tuple[str, ...]:
    patterns: list[str] = []
    if summary is not None:
        patterns.append(
            f"전체 신호는 {summary.signal_level.value}, "
            f"점수 {summary.signal_score:.1f}, "
            f"추세 {summary.trend.value}로 관측됨"
        )
    for note in notes:
        if note.startswith(("반복 주제:", "최근 근거의 주요 카테고리:", "최근 7일")):
            patterns.append(_shorten_context_line(note))
        if len(patterns) >= 5:
            break
    return tuple(patterns)


def _build_inferred_state(
    *,
    notes: tuple[str, ...],
    child_statements: tuple[str, ...],
) -> tuple[str, ...]:
    source = " ".join((*notes, *child_statements))
    inferred: list[str] = []
    if any(marker in source for marker in ("자해/죽음", "죽고", "자살", "자해")):
        inferred.append(
            "죽음이나 자해와 연결된 생각이 커졌을 수 있어 안전 확인이 중요함"
        )
    if any(marker in source for marker in ("우울", "무기력", "힘들", "지쳤")):
        inferred.append("지침, 무기력, 절망감이 함께 올라온 상태일 수 있음")
    if any(marker in source for marker in ("불안", "걱정", "무서", "공포")):
        inferred.append("불안이나 긴장이 높아져 마음이 계속 경계 상태일 수 있음")
    if any(marker in source for marker in ("혼자", "외면", "고립")):
        inferred.append("혼자 버티고 있다는 느낌 때문에 고립감이 강할 수 있음")
    return tuple(inferred[:4])


def _build_open_questions(notes: tuple[str, ...]) -> tuple[str, ...]:
    source = " ".join(notes)
    questions = ["지금 실제로 안전한 장소에 있는지"]
    if "자해/죽음" in source:
        questions.append("곁에 바로 도움을 요청할 수 있는 사람이나 기관이 있는지")
    return tuple(questions)


def _section_lines(title: str, values: tuple[str, ...]) -> list[str]:
    if not values:
        return [f"- {title}: 없음"]
    return [f"- {title}: " + " / ".join(values)]


def _shorten_context_line(value: str) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= _MAX_COUNSELOR_LINE_CHARS:
        return normalized
    return f"{normalized[:_MAX_COUNSELOR_LINE_CHARS].rstrip()}..."
