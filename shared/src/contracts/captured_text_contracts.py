"""브라우저/로컬 collector가 agent로 보내는 captured text 계약."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

CAPTURED_TEXT_EVENT_V1 = "captured_text_event.v1"
CAPTURED_TEXT_INGEST_RESPONSE_V1 = "captured_text_ingest_response.v1"
CAPTURED_TEXT_BATCH_INGEST_RESPONSE_V1 = "captured_text_batch_ingest_response.v1"

CapturedTextEventSchemaVersion: TypeAlias = Literal["captured_text_event.v1"]
CapturedTextIngestResponseSchemaVersion: TypeAlias = Literal[
    "captured_text_ingest_response.v1"
]
CapturedTextBatchIngestResponseSchemaVersion: TypeAlias = Literal[
    "captured_text_batch_ingest_response.v1"
]


class CapturedTextSourceType(StrEnum):
    """captured text를 만든 producer/source 종류."""

    SEARCH = "search"
    REDDIT = "reddit"
    WEBPAGE = "webpage"
    TYPING = "typing"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class CapturedTextSurfaceType(StrEnum):
    """producer가 감지한 구체 입력/콘텐츠 surface."""

    SEARCH_BOX = "search_box"
    ADDRESS_BAR = "address_bar"
    REDDIT_POST = "reddit_post"
    REDDIT_COMMENT = "reddit_comment"
    PAGE_TEXT = "page_text"
    SELECTION = "selection"
    TYPING_SEGMENT = "typing_segment"
    UNKNOWN = "unknown"


class CapturedTextEventPayload(BaseModel):
    """확장 프로그램/로컬 collector가 agent에 보내는 local-only raw text event.

    이 payload는 raw text와 page-adjacent metadata를 포함하므로 agent local
    boundary 안에서만 소비한다. main_server나 FL update envelope으로 전달하지
    않는다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: CapturedTextEventSchemaVersion = CAPTURED_TEXT_EVENT_V1
    event_id: str = Field(min_length=1, max_length=128)
    occurred_at: datetime
    text: str = Field(min_length=1, max_length=20000)
    locale: str = Field(default="ko", min_length=2, max_length=16)
    source_type: CapturedTextSourceType = CapturedTextSourceType.UNKNOWN
    surface_type: CapturedTextSurfaceType = CapturedTextSurfaceType.UNKNOWN
    page_url: str | None = Field(default=None, max_length=2048)
    page_title: str | None = Field(default=None, max_length=512)
    collector_version: str | None = Field(default=None, max_length=128)
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_text(self) -> "CapturedTextEventPayload":
        if not self.text.strip():
            raise ValueError("CapturedTextEvent text must not be blank.")
        return self


class CapturedTextBatchIngestRequestPayload(BaseModel):
    """captured text batch ingest 요청."""

    model_config = ConfigDict(extra="forbid")

    events: tuple[CapturedTextEventPayload, ...] = Field(min_length=1, max_length=100)


class CapturedTextIngestResponsePayload(BaseModel):
    """agent가 단일 captured text event를 처리한 결과."""

    model_config = ConfigDict(extra="forbid")

    schema_version: CapturedTextIngestResponseSchemaVersion = (
        CAPTURED_TEXT_INGEST_RESPONSE_V1
    )
    event_id: str
    query_id: str
    top_category: str | None = None
    top_score: float | None = None
    message: str


class CapturedTextBatchIngestResponsePayload(BaseModel):
    """agent가 batch captured text event를 처리한 결과."""

    model_config = ConfigDict(extra="forbid")

    schema_version: CapturedTextBatchIngestResponseSchemaVersion = (
        CAPTURED_TEXT_BATCH_INGEST_RESPONSE_V1
    )
    processed: int = Field(ge=0)
    results: tuple[CapturedTextIngestResponsePayload, ...]
