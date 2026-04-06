"""Prototype build state 파일 저장소."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shared.src.contracts.prototype_build_state_contracts import (
    SinglePrototypeBuildStatePayload,
    dump_prototype_build_state_payload,
    load_prototype_build_state_payload,
)

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class PrototypeBuildStateRepository:
    """중앙 서버의 versioned prototype build state 파일을 관리한다."""

    state_root: Path = field(
        default_factory=lambda: MAIN_SERVER_ROOT / "state" / "prototype_build_states"
    )

    @property
    def versions_dir(self) -> Path:
        return self.state_root / "versions"

    def save_state(self, payload: SinglePrototypeBuildStatePayload) -> Path:
        path = self.path_for_version(payload.prototype_version)
        dump_prototype_build_state_payload(path, payload)
        return path

    def load_state(self, prototype_version: str) -> SinglePrototypeBuildStatePayload:
        path = self.path_for_version(prototype_version)
        if not path.exists():
            raise FileNotFoundError(
                f"Prototype build state not found: {prototype_version}"
            )
        return load_prototype_build_state_payload(path)

    def has_state(self, prototype_version: str) -> bool:
        return self.path_for_version(prototype_version).exists()

    def path_for_version(self, prototype_version: str) -> Path:
        return self.versions_dir / f"{prototype_version}.json"
