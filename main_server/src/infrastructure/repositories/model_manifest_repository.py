"""서버가 배포하는 model manifest 저장소."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from main_server.src.services.federation.rounds.boundary.payloads import (
    ActiveModelManifestPointerPayload,
    dump_active_model_manifest_pointer_payload,
    load_active_model_manifest_pointer_payload,
)
from shared.src.contracts.model_contracts import (
    ModelManifestPayload,
    dump_model_manifest_payload,
    load_model_manifest_payload,
)

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class ModelManifestRepository:
    """revision별 model manifest와 current 포인터를 관리한다."""

    state_root: Path = field(
        default_factory=lambda: MAIN_SERVER_ROOT / "state" / "model_manifests"
    )

    @property
    def versions_dir(self) -> Path:
        return self.state_root / "versions"

    @property
    def active_pointer_path(self) -> Path:
        return self.state_root / "active.json"

    def path_for_revision(self, model_revision: str) -> Path:
        return self.versions_dir / f"{model_revision}.json"

    def has_manifest(self, model_revision: str) -> bool:
        return self.path_for_revision(model_revision).exists()

    def save_model_manifest(self, payload: ModelManifestPayload) -> Path:
        path = self.path_for_revision(payload.model_revision)
        dump_model_manifest_payload(path, payload)
        return path

    def load_model_manifest(self, model_revision: str) -> ModelManifestPayload:
        path = self.path_for_revision(model_revision)
        if not path.exists():
            raise FileNotFoundError(f"Model manifest not found: {model_revision}")
        return load_model_manifest_payload(path)

    def load_active_pointer(self) -> ActiveModelManifestPointerPayload | None:
        if not self.active_pointer_path.exists():
            return None
        return load_active_model_manifest_pointer_payload(self.active_pointer_path)

    def set_active(
        self,
        model_revision: str,
        *,
        activated_at: datetime,
    ) -> ActiveModelManifestPointerPayload:
        if not self.has_manifest(model_revision):
            raise FileNotFoundError(f"Model manifest not found: {model_revision}")
        pointer = ActiveModelManifestPointerPayload(
            model_revision=model_revision,
            activated_at=activated_at,
        )
        dump_active_model_manifest_pointer_payload(self.active_pointer_path, pointer)
        return pointer

    def load_active_model_manifest(self) -> ModelManifestPayload:
        pointer = self.load_active_pointer()
        if pointer is None:
            raise FileNotFoundError("No active model manifest is registered.")
        return self.load_model_manifest(pointer.model_revision)
