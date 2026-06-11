"""Training example backend 이름 계약."""

from __future__ import annotations

WEAK_STRONG_PAIR_EXAMPLE_BACKEND = "weak_strong_pair"

EXAMPLE_BACKENDS_REQUIRING_WEAK_STRONG_SOURCE_ROWS = frozenset(
    {WEAK_STRONG_PAIR_EXAMPLE_BACKEND}
)


def example_backend_requires_weak_strong_source_rows(backend_name: str) -> bool:
    """backend가 source row에서 weak/strong text pair를 요구하는지 반환한다."""

    normalized = backend_name.strip().lower().replace("-", "_")
    return normalized in EXAMPLE_BACKENDS_REQUIRING_WEAK_STRONG_SOURCE_ROWS
