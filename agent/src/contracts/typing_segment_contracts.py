"""브라우저/로컬 입력 surface에서 생성된 typing segment 계약."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

TYPING_SEGMENT_V1 = "typing_segment.v1"
TYPING_SEGMENT_INGEST_RESPONSE_V1 = "typing_segment_ingest_response.v1"
TYPING_SEGMENT_BATCH_INGEST_RESPONSE_V1 = "typing_segment_batch_ingest_response.v1"

TypingSegmentSchemaVersion: TypeAlias = Literal["typing_segment.v1"]
TypingSegmentIngestResponseSchemaVersion: TypeAlias = Literal[
    "typing_segment_ingest_response.v1"
]
TypingSegmentBatchIngestResponseSchemaVersion: TypeAlias = Literal[
    "typing_segment_batch_ingest_response.v1"
]


class TypingSegmentSourceType(StrEnum):
    """typing segment를 만든 producer 계층."""

    BROWSER_EXTENSION = "browser_extension"
    DESKTOP_APP = "desktop_app"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class TypingSurfaceType(StrEnum):
    """producer가 감지한 입력 surface 종류."""

    INPUT = "input"
    TEXTAREA = "textarea"
    CONTENTEDITABLE = "contenteditable"
    RICH_EDITOR = "rich_editor"
    UNKNOWN = "unknown"


class TypingCaptureConfidence(StrEnum):
    """producer가 segment 복원 품질을 평가한 수준."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TypingSegmentStatsPayload(BaseModel):
    """segment 생성 중 관측된 편집 event 요약."""

    model_config = ConfigDict(extra="forbid")

    insert_count: int = Field(default=0, ge=0)
    delete_count: int = Field(default=0, ge=0)
    paste_count: int = Field(default=0, ge=0)
    composition_count: int = Field(default=0, ge=0)


class TypingSegmentPayload(BaseModel):
    """확장 프로그램/로컬 collector가 agent로 보내는 local-only segment.

    이 payload는 raw text를 포함할 수 있으므로 agent local boundary 안에서만
    소비한다. main_server나 FL update envelope으로 전달하지 않는다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: TypingSegmentSchemaVersion = TYPING_SEGMENT_V1
    segment_id: str = Field(min_length=1)
    source_type: TypingSegmentSourceType = TypingSegmentSourceType.UNKNOWN
    surface_type: TypingSurfaceType = TypingSurfaceType.UNKNOWN
    capture_confidence: TypingCaptureConfidence = TypingCaptureConfidence.MEDIUM
    page_origin: str | None = Field(default=None, max_length=512)
    page_url: str | None = Field(default=None, max_length=2048)
    field_hint: str | None = Field(
        default=None,
        max_length=256,
        description="placeholder/aria-label/name 같은 입력 surface 힌트.",
    )
    started_at: datetime
    ended_at: datetime
    idle_ms: int = Field(ge=0)
    locale: str = Field(default="ko", min_length=2, max_length=16)
    final_text: str | None = Field(default=None, max_length=20000)
    deleted_text: str | None = Field(default=None, max_length=20000)
    stats: TypingSegmentStatsPayload = Field(default_factory=TypingSegmentStatsPayload)

    @model_validator(mode="after")
    def _validate_temporal_order_and_text(self) -> "TypingSegmentPayload":
        if self.ended_at < self.started_at:
            raise ValueError("TypingSegment ended_at must be greater than started_at.")
        if not self.analysis_text:
            raise ValueError(
                "TypingSegment must include final_text or deleted_text for analysis."
            )
        return self

    @property
    def analysis_text(self) -> str:
        """agent inference에 사용할 대표 텍스트."""

        final_text = (self.final_text or "").strip()
        if final_text:
            return final_text
        return (self.deleted_text or "").strip()


class TypingSegmentIngestResponsePayload(BaseModel):
    """agent가 단일 typing segment를 처리한 결과."""

    model_config = ConfigDict(extra="forbid")

    schema_version: TypingSegmentIngestResponseSchemaVersion = (
        TYPING_SEGMENT_INGEST_RESPONSE_V1
    )
    segment_id: str
    query_id: str
    top_category: str | None = None
    top_score: float | None = None
    message: str


class TypingSegmentBatchIngestRequestPayload(BaseModel):
    """typing segment batch ingest 요청."""

    model_config = ConfigDict(extra="forbid")

    segments: tuple[TypingSegmentPayload, ...] = Field(min_length=1, max_length=100)


class TypingSegmentBatchIngestResponsePayload(BaseModel):
    """agent가 batch typing segment를 처리한 결과."""

    model_config = ConfigDict(extra="forbid")

    schema_version: TypingSegmentBatchIngestResponseSchemaVersion = (
        TYPING_SEGMENT_BATCH_INGEST_RESPONSE_V1
    )
    processed: int = Field(ge=0)
    results: tuple[TypingSegmentIngestResponsePayload, ...]

