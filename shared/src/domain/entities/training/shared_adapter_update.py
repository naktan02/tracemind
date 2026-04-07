"""공통 shared adapter update 프로토콜."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class SharedAdapterUpdate(Protocol):
    """공통 shared adapter update가 제공해야 하는 최소 인터페이스."""

    schema_version: str
    adapter_kind: str
    model_id: str
    base_model_revision: str
    training_scope: str
    example_count: int
    created_at: datetime | None
