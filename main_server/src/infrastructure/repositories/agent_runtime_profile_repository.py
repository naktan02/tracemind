"""Server active agent runtime profile 저장소."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shared.src.contracts.agent_runtime_profile_contracts import (
    AgentRuntimeProfilePayload,
)

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class AgentRuntimeProfileRepository:
    """active agent runtime profile과 history를 파일로 관리한다."""

    state_root: Path = field(
        default_factory=lambda: MAIN_SERVER_ROOT / "state" / "agent_runtime_profiles"
    )

    @property
    def active_path(self) -> Path:
        return self.state_root / "active.json"

    @property
    def history_dir(self) -> Path:
        return self.state_root / "history"

    def save_active(
        self,
        profile: AgentRuntimeProfilePayload,
    ) -> AgentRuntimeProfilePayload:
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        payload = profile.model_dump_json(indent=2) + "\n"
        self.active_path.write_text(payload, encoding="utf-8")
        history_path = self.history_dir / (
            f"{profile.profile_id}__{profile.profile_revision}.json"
        )
        history_path.write_text(payload, encoding="utf-8")
        return profile

    def load_active(self) -> AgentRuntimeProfilePayload:
        if not self.active_path.exists():
            raise FileNotFoundError("No active agent runtime profile is configured.")
        return AgentRuntimeProfilePayload.model_validate_json(
            self.active_path.read_text(encoding="utf-8")
        )
