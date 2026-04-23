"""FL round lifecycle 상태 저장소."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from main_server.src.services.federation.rounds.boundary.mappers import (
    round_record_from_payload,
    round_record_to_payload,
)
from main_server.src.services.federation.rounds.boundary.models import RoundRecord
from main_server.src.services.federation.rounds.boundary.payloads import (
    ActiveRoundPointerPayload,
    dump_active_round_pointer_payload,
    dump_round_record_payload,
    load_active_round_pointer_payload,
    load_round_record_payload,
)

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class RoundRepository:
    """중앙 서버의 round state 파일을 관리한다."""

    state_root: Path = field(
        default_factory=lambda: MAIN_SERVER_ROOT / "state" / "rounds"
    )

    @property
    def rounds_dir(self) -> Path:
        return self.state_root / "records"

    @property
    def active_pointer_path(self) -> Path:
        return self.state_root / "active.json"

    def path_for_round(self, round_id: str) -> Path:
        return self.rounds_dir / f"{round_id}.json"

    def has_round(self, round_id: str) -> bool:
        return self.path_for_round(round_id).exists()

    def save_round(self, record: RoundRecord) -> Path:
        path = self.path_for_round(record.round_id)
        dump_round_record_payload(path, round_record_to_payload(record))
        return path

    def load_round(self, round_id: str) -> RoundRecord:
        path = self.path_for_round(round_id)
        if not path.exists():
            raise FileNotFoundError(f"Round not found: {round_id}")
        return round_record_from_payload(load_round_record_payload(path))

    def load_active_pointer(self) -> ActiveRoundPointerPayload | None:
        if not self.active_pointer_path.exists():
            return None
        return load_active_round_pointer_payload(self.active_pointer_path)

    def set_active(
        self,
        round_id: str,
        *,
        activated_at: datetime,
    ) -> ActiveRoundPointerPayload:
        if not self.has_round(round_id):
            raise FileNotFoundError(f"Round not found: {round_id}")
        pointer = ActiveRoundPointerPayload(
            round_id=round_id,
            activated_at=activated_at,
        )
        dump_active_round_pointer_payload(self.active_pointer_path, pointer)
        return pointer

    def clear_active(self, *, expected_round_id: str | None = None) -> None:
        if not self.active_pointer_path.exists():
            return
        if expected_round_id is not None:
            active = self.load_active_pointer()
            if active is None or active.round_id != expected_round_id:
                return
        self.active_pointer_path.unlink()
