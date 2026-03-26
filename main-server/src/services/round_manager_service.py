"""FL round 관리자 자리표시자."""

from dataclasses import dataclass


@dataclass(slots=True)
class RoundManagerService:
    """향후 선택적으로 열릴 FL round를 조정한다."""

    enabled: bool = False
