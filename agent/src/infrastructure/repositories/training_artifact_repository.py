"""Agent 로컬 학습 artifact 저장소."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shared.src.contracts.adapter_contracts import (
    VectorAdapterDeltaPayload,
    dump_vector_adapter_delta_payload,
    load_vector_adapter_delta_payload,
)

AGENT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class TrainingArtifactRepository:
    """로컬 update payload를 버전형 파일로 저장한다."""

    state_root: Path = field(
        default_factory=lambda: AGENT_ROOT / "state" / "training_updates"
    )

    @property
    def delta_dir(self) -> Path:
        return self.state_root / "vector_adapter_deltas"

    def path_for_update(self, update_id: str) -> Path:
        return self.delta_dir / f"{update_id}.json"

    def save_vector_adapter_delta(
        self,
        update_id: str,
        payload: VectorAdapterDeltaPayload,
    ) -> Path:
        path = self.path_for_update(update_id)
        dump_vector_adapter_delta_payload(path, payload)
        return path

    def load_vector_adapter_delta(self, update_id: str) -> VectorAdapterDeltaPayload:
        path = self.path_for_update(update_id)
        if not path.exists():
            raise FileNotFoundError(f"Training delta not found: {update_id}")
        return load_vector_adapter_delta_payload(path)
