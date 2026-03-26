"""개인 기준선 프로필."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BaselineProfile:
    """한 로컬 사용자에 대한 rolling self-baseline feature."""

    profile_version: str
    warmup_complete: bool
    category_means: dict[str, float] = field(default_factory=dict)
    category_sigmas: dict[str, float] = field(default_factory=dict)
    persistence_days: int = 0
