"""아이용 LLM 컨텍스트에 넣을 최근 캡처/검색 근거 요약."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from agent.src.features.captured_text.storage.records import CapturedTextRecord
from agent.src.features.captured_text.storage.repository import CapturedTextRepository
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from shared.src.domain.entities.inference.events import AnalysisEvent

_NORMAL_CATEGORY = "normal"
_TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9]{1,}")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NUMBER_PATTERN = re.compile(r"\d+")

_STOPWORDS = frozenset(
    {
        "그리고",
        "그래서",
        "그런데",
        "하지만",
        "나는",
        "내가",
        "너무",
        "진짜",
        "오늘",
        "지금",
        "계속",
        "그냥",
        "really",
        "about",
        "with",
        "from",
        "this",
        "that",
        "what",
        "when",
        "where",
        "how",
    }
)

_TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("자해/죽음 관련 표현", ("죽고", "자해", "극단", "suicide", "selfharm")),
    ("불안/공포", ("불안", "걱정", "무서", "공포", "panic", "anxiety")),
    ("우울/무기력", ("우울", "무기력", "외롭", "힘들", "depress", "lonely")),
    ("친구/관계 갈등", ("친구", "따돌", "괴롭", "관계", "싸움", "bully")),
    ("가족/보호자 갈등", ("가족", "부모", "엄마", "아빠", "보호자")),
    ("학교/성적 부담", ("학교", "시험", "성적", "공부", "숙제")),
    ("수면/생활 리듬", ("잠", "수면", "불면", "sleep")),
    ("신체/식사 걱정", ("몸", "체중", "먹기", "식사", "살", "다이어트")),
)
_HIGH_RISK_TOPIC_LABEL = "자해/죽음 관련 표현"


@dataclass(frozen=True, slots=True)
class ChildSupportEvidenceSummaryConfig:
    """최근 근거를 LLM 컨텍스트용 요약으로 만들 때 쓰는 선택 기준."""

    lookback_days: int = 7
    candidate_limit: int = 16
    min_relevant_score: float = 35.0
    category_limit: int = 4
    topic_limit: int = 6
    keyword_limit: int = 8
    snippet_limit: int = 4
    snippet_max_chars: int = 140


@dataclass(frozen=True, slots=True)
class ChildSupportEvidenceItem:
    """요약기가 읽는 단일 분석 근거."""

    source_event_id: str
    occurred_at: datetime
    category_scores: dict[str, float]
    text: str


@dataclass(frozen=True, slots=True)
class ChildSupportEvidenceSummary:
    """원문을 직접 노출하지 않는 최근 캡처/검색 근거 요약."""

    observed_event_count: int
    selected_event_count: int
    top_categories: tuple[str, ...] = field(default_factory=tuple)
    topics: tuple[str, ...] = field(default_factory=tuple)
    keywords: tuple[str, ...] = field(default_factory=tuple)
    snippets: tuple[str, ...] = field(default_factory=tuple)

    def to_prompt_lines(self) -> tuple[str, ...]:
        if self.selected_event_count == 0:
            return ()
        evidence_count = (
            f"{self.observed_event_count}건 중 관련도 높은 "
            f"{self.selected_event_count}건"
        )
        lines = [(f"최근 캡처/검색 근거: {evidence_count}을 원문 없이 요약함")]
        if self.top_categories:
            lines.append("최근 근거의 주요 카테고리: " + ", ".join(self.top_categories))
        if self.topics:
            lines.append("반복 주제: " + ", ".join(self.topics))
        if self.keywords:
            lines.append("반복 표현 힌트: " + ", ".join(self.keywords))
        if self.snippets:
            lines.append("최근 원문 일부: " + " / ".join(self.snippets))
        return tuple(lines)


class ChildSupportEvidenceSummarizer(Protocol):
    """최근 분석 이벤트를 LLM prompt용 요약으로 바꾸는 교체 지점."""

    def summarize(
        self,
        items: Sequence[ChildSupportEvidenceItem],
        *,
        observed_event_count: int,
    ) -> ChildSupportEvidenceSummary:
        """원문을 직접 보존하지 않는 근거 요약을 만든다."""


@dataclass(slots=True)
class HeuristicChildSupportEvidenceSummarizer:
    """LLM 없이 동작하는 기본 근거 요약기."""

    config: ChildSupportEvidenceSummaryConfig = field(
        default_factory=ChildSupportEvidenceSummaryConfig
    )

    def summarize(
        self,
        items: Sequence[ChildSupportEvidenceItem],
        *,
        observed_event_count: int,
    ) -> ChildSupportEvidenceSummary:
        selected = list(items[: self.config.candidate_limit])
        category_counter: Counter[str] = Counter()
        topic_counter: Counter[str] = Counter()
        token_counter: Counter[str] = Counter()

        for item in selected:
            for category, score in item.category_scores.items():
                if category == _NORMAL_CATEGORY:
                    continue
                if float(score) >= self.config.min_relevant_score:
                    category_counter[category] += 1
            normalized_text = _sanitize_for_local_summary(item.text)
            lowered = normalized_text.lower()
            for label, keywords in _TOPIC_KEYWORDS:
                if any(keyword in lowered for keyword in keywords):
                    topic_counter[label] += 1
            token_counter.update(_extract_tokens(normalized_text))

        return ChildSupportEvidenceSummary(
            observed_event_count=observed_event_count,
            selected_event_count=len(selected),
            top_categories=tuple(
                category
                for category, _ in category_counter.most_common(
                    self.config.category_limit
                )
            ),
            topics=tuple(
                topic for topic, _ in topic_counter.most_common(self.config.topic_limit)
            ),
            keywords=tuple(
                token
                for token, _ in token_counter.most_common(self.config.keyword_limit)
            ),
            snippets=tuple(
                _shorten_snippet(_sanitize_for_local_summary(item.text), self.config)
                for item in selected[: self.config.snippet_limit]
                if _sanitize_for_local_summary(item.text)
            ),
        )


@dataclass(slots=True)
class ChildSupportEvidenceSummaryBuilder:
    """analysis event와 captured text를 읽어 최근 근거 요약을 만든다."""

    analysis_event_repository: AnalysisEventRepository
    captured_text_repository: CapturedTextRepository | None = None
    summarizer: ChildSupportEvidenceSummarizer = field(
        default_factory=HeuristicChildSupportEvidenceSummarizer
    )
    config: ChildSupportEvidenceSummaryConfig = field(
        default_factory=ChildSupportEvidenceSummaryConfig
    )

    def build(self) -> ChildSupportEvidenceSummary:
        events = self.analysis_event_repository.get_recent(
            days=self.config.lookback_days
        )
        items = [
            ChildSupportEvidenceItem(
                source_event_id=event.source_event_id or event.query_id,
                occurred_at=event.occurred_at,
                category_scores=dict(event.category_scores),
                text=self._load_text(event),
            )
            for event in events
        ]
        selected_items = _select_relevant_items(
            items,
            min_relevant_score=self.config.min_relevant_score,
            limit=self.config.candidate_limit,
        )
        return self.summarizer.summarize(
            selected_items,
            observed_event_count=len(events),
        )

    def _load_text(self, event: AnalysisEvent) -> str:
        source_event_id = event.source_event_id or event.query_id
        if self.captured_text_repository is not None:
            try:
                record = self.captured_text_repository.get(source_event_id)
            except Exception:
                record = None
            if isinstance(record, CapturedTextRecord):
                return record.text
        return event.translated_text or ""


def _select_relevant_items(
    items: Sequence[ChildSupportEvidenceItem],
    *,
    min_relevant_score: float,
    limit: int,
) -> list[ChildSupportEvidenceItem]:
    scored = sorted(
        items,
        key=lambda item: (
            _contains_high_risk_topic(item.text),
            _max_non_normal_score(item.category_scores),
            item.occurred_at,
        ),
        reverse=True,
    )
    relevant = [
        item
        for item in scored
        if _max_non_normal_score(item.category_scores) >= min_relevant_score
        or _contains_high_risk_topic(item.text)
    ]
    if not relevant:
        relevant = scored
    return relevant[:limit]


def _max_non_normal_score(category_scores: dict[str, float]) -> float:
    return max(
        (
            float(score)
            for category, score in category_scores.items()
            if category != _NORMAL_CATEGORY
        ),
        default=0.0,
    )


def _contains_high_risk_topic(text: str) -> bool:
    lowered = text.lower()
    for label, keywords in _TOPIC_KEYWORDS:
        if label != _HIGH_RISK_TOPIC_LABEL:
            continue
        return any(keyword in lowered for keyword in keywords)
    return False


def _sanitize_for_local_summary(text: str) -> str:
    text = _URL_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)
    text = _NUMBER_PATTERN.sub(" ", text)
    return " ".join(text.split())


def _shorten_snippet(
    text: str,
    config: ChildSupportEvidenceSummaryConfig,
) -> str:
    if len(text) <= config.snippet_max_chars:
        return text
    return f"{text[: config.snippet_max_chars].rstrip()}..."


def _extract_tokens(text: str) -> list[str]:
    tokens = []
    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0).lower()
        if token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens
