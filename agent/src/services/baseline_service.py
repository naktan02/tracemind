"""개인 기준선 서비스 자리표시자."""

from dataclasses import dataclass


@dataclass(slots=True)
class BaselineService:
    """개인 기준선과 지속성 feature를 계산한다."""

    warmup_days: int = 28
