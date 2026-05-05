"""Agent 로컬 PrototypePack 캐시 저장소."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from shared.src.contracts.prototype_contracts import (
    PrototypePackActivationPointer,
    PrototypePackPayload,
    dump_activation_pointer,
    dump_prototype_pack_payload,
    load_activation_pointer,
    load_prototype_pack_payload,
)

AGENT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class PrototypePackRepository:
    """agent의 로컬 prototype pack 버전 캐시와 active pointer를 관리한다."""

    state_root: Path = field(
        default_factory=lambda: AGENT_ROOT / "state" / "prototype_packs"
    )

    @property
    def versions_dir(self) -> Path:
        return self.state_root / "versions"

    @property
    def active_pointer_path(self) -> Path:
        return self.state_root / "active.json"

    def save_pack(self, payload: PrototypePackPayload) -> Path:
        path = self.path_for_version(payload.prototype_version)
        dump_prototype_pack_payload(path, payload)
        return path

    def load_pack(self, prototype_version: str) -> PrototypePackPayload:
        path = self.path_for_version(prototype_version)
        if not path.exists():
            raise FileNotFoundError(f"Prototype pack not found: {prototype_version}")
        return load_prototype_pack_payload(path)

    def path_for_version(self, prototype_version: str) -> Path:
        return self.versions_dir / f"{prototype_version}.json"

    def load_active_pointer(self) -> PrototypePackActivationPointer | None:
        if not self.active_pointer_path.exists():
            return None
        return load_activation_pointer(self.active_pointer_path)

    def set_active(self, prototype_version: str) -> PrototypePackActivationPointer:
        if not self.path_for_version(prototype_version).exists():
            raise FileNotFoundError(f"Prototype pack not found: {prototype_version}")

        pointer = PrototypePackActivationPointer(
            prototype_version=prototype_version,
            activated_at=datetime.now(timezone.utc),
        )
        dump_activation_pointer(self.active_pointer_path, pointer)
        return pointer
