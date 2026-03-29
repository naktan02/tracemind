"""전역 vector adapter 상태 저장소."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shared.src.contracts.adapter_contracts import (
    VectorAdapterStatePayload,
    dump_vector_adapter_state_payload,
    load_vector_adapter_state_payload,
)

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class VectorAdapterStateRepository:
    """revision별 vector adapter 상태 파일을 관리한다."""

    state_root: Path = field(
        default_factory=lambda: MAIN_SERVER_ROOT / "state" / "vector_adapter_states"
    )

    @property
    def versions_dir(self) -> Path:
        return self.state_root / "versions"

    def path_for_revision(self, model_revision: str) -> Path:
        return self.versions_dir / f"{model_revision}.json"

    def save_state(self, payload: VectorAdapterStatePayload) -> Path:
        path = self.path_for_revision(payload.model_revision)
        dump_vector_adapter_state_payload(path, payload)
        return path

    def load_state(self, model_revision: str) -> VectorAdapterStatePayload:
        path = self.path_for_revision(model_revision)
        if not path.exists():
            raise FileNotFoundError(f"Vector adapter state not found: {model_revision}")
        return load_vector_adapter_state_payload(path)

    def load_state_from_ref(self, artifact_ref: str) -> VectorAdapterStatePayload:
        path = Path(artifact_ref)
        if not path.exists():
            raise FileNotFoundError(
                f"Vector adapter state path not found: {artifact_ref}"
            )
        return load_vector_adapter_state_payload(path)
