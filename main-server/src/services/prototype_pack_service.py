"""Prototype pack 배포 서비스 자리표시자."""

from dataclasses import dataclass


@dataclass(slots=True)
class PrototypePackService:
    """labeled query set에서 만든 semantic prototype pack을 배포한다."""

    pack_version: str = "bootstrap"
