"""TimingRecorder 단위 검증."""

from __future__ import annotations

from methods.common.timing import TimingRecorder


def test_timing_recorder_accumulates_named_sections() -> None:
    recorder = TimingRecorder()

    with recorder.measure("section_seconds"):
        pass
    with recorder.measure("section_seconds"):
        pass

    payload = recorder.to_mapping()

    assert set(payload) == {"section_seconds"}
    assert payload["section_seconds"] >= 0.0
