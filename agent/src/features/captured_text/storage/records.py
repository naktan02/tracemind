"""Captured text repository record objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from agent.src.contracts.captured_text_contracts import (
    CAPTURED_TEXT_EVENT_V1,
    CapturedTextEventPayload,
)

CAPTURED_TEXT_VIEW_STATUS_PENDING = "pending"
CAPTURED_TEXT_VIEW_STATUS_DUPLICATE = "duplicate"
CAPTURED_TEXT_VIEW_STATUS_READY = "ready"
CAPTURED_TEXT_VIEW_STATUS_FAILED = "failed"
CAPTURED_TEXT_ANALYSIS_STATUS_PENDING = "pending"
CAPTURED_TEXT_ANALYSIS_STATUS_COMPLETED = "completed"
CAPTURED_TEXT_ANALYSIS_STATUS_FAILED = "failed"
CAPTURED_TEXT_GENERATED_VIEW_V1 = "captured_text_generated_view.v1"

CAPTURED_TEXT_VIEW_STATUSES = frozenset(
    {
        CAPTURED_TEXT_VIEW_STATUS_PENDING,
        CAPTURED_TEXT_VIEW_STATUS_DUPLICATE,
        CAPTURED_TEXT_VIEW_STATUS_READY,
        CAPTURED_TEXT_VIEW_STATUS_FAILED,
    }
)

CAPTURED_TEXT_ANALYSIS_STATUSES = frozenset(
    {
        CAPTURED_TEXT_ANALYSIS_STATUS_PENDING,
        CAPTURED_TEXT_ANALYSIS_STATUS_COMPLETED,
        CAPTURED_TEXT_ANALYSIS_STATUS_FAILED,
    }
)


@dataclass(slots=True)
class CapturedTextRecord:
    """agent 로컬 captured text raw event snapshot."""

    event_id: str
    occurred_at: datetime
    received_at: datetime
    text: str
    locale: str
    source_type: str
    surface_type: str
    page_url: str | None = None
    page_title: str | None = None
    collector_version: str | None = None
    text_fingerprint: str = ""
    duplicate_of_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CAPTURED_TEXT_EVENT_V1

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty.")
        if not self.text.strip():
            raise ValueError("text must not be empty.")
        if not self.locale.strip():
            raise ValueError("locale must not be empty.")
        if not self.source_type.strip():
            raise ValueError("source_type must not be empty.")
        if not self.surface_type.strip():
            raise ValueError("surface_type must not be empty.")
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty.")
        if self.duplicate_of_event_id is not None and not (
            self.duplicate_of_event_id.strip()
        ):
            raise ValueError("duplicate_of_event_id must not be empty.")


@dataclass(slots=True)
class CapturedTextGeneratedViewRecord:
    """agent-local captured text에서 만든 weak/strong view snapshot."""

    event_id: str
    generated_at: datetime
    weak_text: str
    strong_text_0: str
    strong_text_1: str
    generator_name: str
    generator_version: str
    source_text_fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CAPTURED_TEXT_GENERATED_VIEW_V1

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty.")
        if not self.weak_text.strip():
            raise ValueError("weak_text must not be empty.")
        if not self.strong_text_0.strip():
            raise ValueError("strong_text_0 must not be empty.")
        if not self.strong_text_1.strip():
            raise ValueError("strong_text_1 must not be empty.")
        if not self.generator_name.strip():
            raise ValueError("generator_name must not be empty.")
        if not self.generator_version.strip():
            raise ValueError("generator_version must not be empty.")
        if not self.source_text_fingerprint.strip():
            raise ValueError("source_text_fingerprint must not be empty.")
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty.")


@dataclass(slots=True)
class CapturedTextGeneratedTrainingSourceRecord:
    """generated view와 원본 captured text를 합친 학습 source snapshot."""

    event_id: str
    occurred_at: datetime
    text: str
    locale: str
    source_type: str
    surface_type: str
    text_fingerprint: str
    generated_at: datetime
    weak_text: str
    strong_text_0: str
    strong_text_1: str
    generator_name: str
    generator_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CapturedTextAnalysisSourceRecord:
    """analysis job이 읽는 generated view + 원본 event snapshot."""

    event_id: str
    occurred_at: datetime
    text: str
    locale: str
    source_type: str
    surface_type: str
    text_fingerprint: str
    generated_at: datetime
    weak_text: str
    strong_text_0: str
    strong_text_1: str
    generator_name: str
    generator_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


def captured_text_record_from_payload(
    payload: CapturedTextEventPayload,
    *,
    received_at: datetime | None = None,
) -> CapturedTextRecord:
    """CapturedTextEventPayload를 저장소 record로 정규화한다."""

    return CapturedTextRecord(
        event_id=payload.event_id,
        schema_version=payload.schema_version,
        occurred_at=payload.occurred_at,
        received_at=received_at or datetime.now(tz=timezone.utc),
        text=payload.text,
        locale=payload.locale,
        source_type=payload.source_type.value,
        surface_type=payload.surface_type.value,
        page_url=payload.page_url,
        page_title=payload.page_title,
        collector_version=payload.collector_version,
        text_fingerprint=text_fingerprint(
            text=payload.text,
            locale=payload.locale,
            source_type=payload.source_type.value,
            surface_type=payload.surface_type.value,
        ),
        metadata=dict(payload.metadata),
    )


def record_payload(record: CapturedTextRecord) -> dict[str, Any]:
    return {
        "event_id": record.event_id,
        "schema_version": record.schema_version,
        "occurred_at": record.occurred_at,
        "received_at": record.received_at,
        "text": record.text,
        "locale": record.locale,
        "source_type": record.source_type,
        "surface_type": record.surface_type,
        "page_url": record.page_url,
        "page_title": record.page_title,
        "collector_version": record.collector_version,
        "metadata": dict(record.metadata),
    }


def text_fingerprint(
    *,
    text: str,
    locale: str,
    source_type: str,
    surface_type: str,
) -> str:
    payload = {
        "locale": locale.strip().lower(),
        "source_type": source_type.strip().lower(),
        "surface_type": surface_type.strip().lower(),
        "text": " ".join(text.strip().lower().split()),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return sha256(encoded.encode("utf-8")).hexdigest()
