"""서버가 수락한 shared adapter update payload 저장소."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shared.src.contracts.adapter_contracts import (
    SharedAdapterUpdatePayload,
    dump_shared_adapter_update_payload,
    load_shared_adapter_update_payload,
)

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[3]
SHARED_ADAPTER_UPDATE_REF_PREFIX = "shared_adapter_update::"


@dataclass(slots=True)
class SharedAdapterUpdateRepository:
    """서버 소유 update payload 파일을 update_id 기준으로 관리한다."""

    state_root: Path = field(
        default_factory=lambda: MAIN_SERVER_ROOT / "state" / "shared_adapter_updates"
    )

    @property
    def versions_dir(self) -> Path:
        return self.state_root / "versions"

    def path_for_update(self, update_id: str) -> Path:
        return self.versions_dir / f"{update_id}.json"

    def ref_for_update(self, update_id: str) -> str:
        """server-owned shared adapter update 참조를 만든다."""

        return f"{SHARED_ADAPTER_UPDATE_REF_PREFIX}{update_id}"

    def update_id_from_ref(self, payload_ref: str) -> str | None:
        """opaque update ref에서 update_id를 추출한다."""

        if not payload_ref.startswith(SHARED_ADAPTER_UPDATE_REF_PREFIX):
            return None
        return payload_ref.removeprefix(SHARED_ADAPTER_UPDATE_REF_PREFIX)

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

    def load_shared_adapter_update_from_ref(
        self,
        payload_ref: str,
    ) -> SharedAdapterUpdatePayload:
        update_id = self.update_id_from_ref(payload_ref)
        if update_id is not None:
            return self.load_shared_adapter_update(update_id)

        # Legacy compatibility: 기존 envelope이 파일 경로를 직접 담고 있을 수 있다.
        path = Path(payload_ref)
        if not path.exists():
            raise FileNotFoundError(
                f"Shared adapter update ref not found: {payload_ref}"
            )
        return load_shared_adapter_update_payload(path)
