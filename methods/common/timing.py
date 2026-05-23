"""경량 wall-clock timing recorder."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class TimingRecorder:
    """구간별 누적 실행 시간을 초 단위로 기록한다."""

    values: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, key: str) -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started_at
            self.values[key] = self.values.get(key, 0.0) + elapsed

    def to_mapping(self) -> dict[str, float]:
        return dict(sorted(self.values.items()))


def timing_mapping(recorder: TimingRecorder | None) -> Mapping[str, float]:
    if recorder is None:
        return {}
    return recorder.to_mapping()
