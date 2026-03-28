"""PrototypePackService unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_SERVER_ROOT))

from shared.src.contracts.prototype_contracts import PrototypePackPayload
from src.infrastructure.repositories.prototype_pack_repository import (
    PrototypePackRepository,
)
from src.services.prototype_pack_service import PrototypePackService


def _load_fixture_payload() -> PrototypePackPayload:
    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "agent"
        / "tests"
        / "unit"
        / "fixtures"
        / "prototype_pack_v1.json"
    )
    return PrototypePackPayload.model_validate_json(fixture_path.read_text(encoding="utf-8"))


def test_publish_activate_and_get_current(tmp_path: Path) -> None:
    repository = PrototypePackRepository(state_root=tmp_path / "prototype_packs")
    service = PrototypePackService(repository=repository)
    payload = _load_fixture_payload()

    pack_path = service.publish_pack(payload)
    active_pointer = service.activate(payload.prototype_version)
    current = service.get_current()

    assert pack_path.exists()
    assert active_pointer.prototype_version == payload.prototype_version
    assert current.pack.prototype_version == payload.prototype_version
    assert sorted(current.pack.categories) == ["anxiety", "depression", "normal", "suicidal"]
