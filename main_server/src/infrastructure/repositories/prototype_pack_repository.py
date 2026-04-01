"""PrototypePack 파일 저장소."""

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

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class PrototypePackRepository:
    """중앙 서버의 versioned prototype pack 파일을 관리한다."""

    state_root: Path = field(
        default_factory=lambda: MAIN_SERVER_ROOT / "state" / "prototype_packs"
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

    def has_pack(self, prototype_version: str) -> bool:
        return self.path_for_version(prototype_version).exists()

    def path_for_version(self, prototype_version: str) -> Path:
        return self.versions_dir / f"{prototype_version}.json"

    def load_active_pointer(self) -> PrototypePackActivationPointer | None:
        if not self.active_pointer_path.exists():
            return None
        return load_activation_pointer(self.active_pointer_path)

    def set_active(self, prototype_version: str) -> PrototypePackActivationPointer:
        if not self.has_pack(prototype_version):
            raise FileNotFoundError(f"Prototype pack not found: {prototype_version}")

        pointer = PrototypePackActivationPointer(
            prototype_version=prototype_version,
            activated_at=datetime.now(timezone.utc),
        )
        dump_activation_pointer(self.active_pointer_path, pointer)
        return pointer
