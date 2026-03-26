"""요약 수집 서비스 자리표시자."""

from dataclasses import dataclass


@dataclass(slots=True)
class IngestionService:
    """프라이버시 안전한 window summary를 수신하고 검증한다."""

    schema_version: str = "window_summary.v1"
