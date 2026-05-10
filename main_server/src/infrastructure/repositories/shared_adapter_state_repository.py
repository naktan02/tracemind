"""전역 shared adapter 상태 저장소."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterStatePayload,
)
from shared.src.contracts.adapter_contract_families.io import (
    dump_shared_adapter_state_payload,
    load_shared_adapter_state_payload,
)

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[3]
SHARED_ADAPTER_STATE_REF_PREFIX = "shared_adapter_state::"


@dataclass(slots=True)
class SharedAdapterStateRepository:
    """revision별 shared adapter 상태 파일을 관리한다."""

    state_root: Path = field(
        default_factory=lambda: MAIN_SERVER_ROOT / "state" / "shared_adapter_states"
    )

    @property
    def versions_dir(self) -> Path:
        return self.state_root / "versions"

    def path_for_revision(self, model_revision: str) -> Path:
        return self.versions_dir / f"{model_revision}.json"

    def ref_for_revision(self, model_revision: str) -> str:
        """server-owned shared adapter state 참조를 만든다."""

        return f"{SHARED_ADAPTER_STATE_REF_PREFIX}{model_revision}"

    def revision_from_ref(self, artifact_ref: str) -> str | None:
        """opaque state ref에서 revision을 추출한다."""

        if not artifact_ref.startswith(SHARED_ADAPTER_STATE_REF_PREFIX):
            return None
        return artifact_ref.removeprefix(SHARED_ADAPTER_STATE_REF_PREFIX)

    def save_shared_adapter_state(self, payload: SharedAdapterStatePayload) -> Path:
        path = self.path_for_revision(payload.model_revision)
        dump_shared_adapter_state_payload(path, payload)
        return path

    def load_shared_adapter_state(
        self,
        model_revision: str,
    ) -> SharedAdapterStatePayload:
        path = self.path_for_revision(model_revision)
        if not path.exists():
            raise FileNotFoundError(f"Shared adapter state not found: {model_revision}")
        return load_shared_adapter_state_payload(path)

    def load_shared_adapter_state_from_ref(
        self,
        artifact_ref: str,
    ) -> SharedAdapterStatePayload:
        revision = self.revision_from_ref(artifact_ref)
        if revision is not None:
            return self.load_shared_adapter_state(revision)

        # Legacy compatibility: 기존 manifest가 파일 경로를 직접 담고 있을 수 있다.
        path = Path(artifact_ref)
        if not path.exists():
            raise FileNotFoundError(
                f"Shared adapter state ref/path not found: {artifact_ref}"
            )
        return load_shared_adapter_state_payload(path)

    def save_state(self, payload: SharedAdapterStatePayload) -> Path:
        return self.save_shared_adapter_state(payload)

    def load_state(self, model_revision: str) -> SharedAdapterStatePayload:
        return self.load_shared_adapter_state(model_revision)

    def load_state_from_ref(self, artifact_ref: str) -> SharedAdapterStatePayload:
        return self.load_shared_adapter_state_from_ref(artifact_ref)


VectorAdapterStateRepository = SharedAdapterStateRepository
