"""Query SSL task identity helpers."""

from __future__ import annotations

from hashlib import sha256

from shared.src.contracts.training_contracts import TrainingTask


def seed_from_task(training_task: TrainingTask) -> int:
    """round/task id에서 deterministic local training seed를 만든다."""

    source = f"{training_task.round_id}:{training_task.task_id}".encode("utf-8")
    return int.from_bytes(sha256(source).digest()[:4], byteorder="big") % (2**31)


def optional_method_name(value: object) -> str | None:
    """task의 optional method-like 값을 빈 문자열 없이 정규화한다."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
