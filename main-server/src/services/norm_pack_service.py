"""Norm pack 배포 서비스 자리표시자."""

from dataclasses import dataclass


@dataclass(slots=True)
class NormPackService:
    """cohort norm pack을 생성하고 배포한다."""

    pack_version: str = "bootstrap"
