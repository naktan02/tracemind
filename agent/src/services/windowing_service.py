"""윈도우 요약 서비스 자리표시자."""

from dataclasses import dataclass


@dataclass(slots=True)
class WindowingService:
    """프라이버시를 보존하는 micro-batch 또는 rolling window 요약을 만든다."""

    window_days: int = 7
