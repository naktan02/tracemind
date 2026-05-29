"""Agent 로컬 학습용 shared adapter update 저장소."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.io import (
    dump_shared_adapter_update_payload,
    load_shared_adapter_update_payload,
)

AGENT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class TrainingArtifactRepository:
    """로컬 shared adapter update payload를 버전형 파일로 저장한다."""

    state_root: Path = field(default_factory=lambda: AGENT_ROOT / "state")

    @property
    def shared_update_dir(self) -> Path:
        return self.state_root / "shared_adapter_updates"

    def path_for_update(self, update_id: str) -> Path:
        return self.shared_update_dir / f"{update_id}.json"

    def save_shared_adapter_update(
        self,
        update_id: str,
        payload: SharedAdapterUpdatePayload,
    ) -> Path:
        path = self.path_for_update(update_id)
        dump_shared_adapter_update_payload(path, payload)
        return path

    def load_shared_adapter_update(self, update_id: str) -> SharedAdapterUpdatePayload:
        path = self.path_for_update(update_id)
        if not path.exists():
            raise FileNotFoundError(f"Shared adapter update not found: {update_id}")
        return load_shared_adapter_update_payload(path)
