"""로컬 판단 서비스 자리표시자."""

from dataclasses import dataclass


@dataclass(slots=True)
class DecisionService:
    """개인 기준선과 또래 기준을 결합해 로컬 판단 결과를 만든다."""

    policy_version: str = "bootstrap"
