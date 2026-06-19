"""Agent 로컬 shared adapter state 캐시 저장소."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agent.src.config.paths import DEFAULT_AGENT_STATE_DIR
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterStatePayload,
)
from shared.src.contracts.adapter_contract_families.io import (
    dump_shared_adapter_state_payload,
    load_shared_adapter_state_payload,
)
from shared.src.contracts.model_contracts import (
    ModelManifestPayload,
    dump_model_manifest_payload,
    load_model_manifest_payload,
)


class SharedAdapterStateActivationPointer(BaseModel):
    """Agent local active shared adapter state pointer."""

    model_config = ConfigDict(extra="forbid")

    model_revision: str
    activated_at: datetime


@dataclass(slots=True)
class SharedAdapterStateRepository:
    """agent의 shared adapter state 버전 캐시와 active pointer를 관리한다."""

    state_root: Path = field(
        default_factory=lambda: DEFAULT_AGENT_STATE_DIR / "shared_adapter_states"
    )

    @property
    def versions_dir(self) -> Path:
        return self.state_root / "versions"

    @property
    def manifests_dir(self) -> Path:
        return self.state_root / "manifests"

    @property
    def active_pointer_path(self) -> Path:
        return self.state_root / "active.json"

    def path_for_revision(self, model_revision: str) -> Path:
        return self.versions_dir / f"{model_revision}.json"

    def manifest_path_for_revision(self, model_revision: str) -> Path:
        return self.manifests_dir / f"{model_revision}.json"

    def save_state(self, payload: SharedAdapterStatePayload) -> Path:
        path = self.path_for_revision(payload.model_revision)
        dump_shared_adapter_state_payload(path, payload)
        return path

    def load_state(self, model_revision: str) -> SharedAdapterStatePayload:
        path = self.path_for_revision(model_revision)
        if not path.exists():
            raise FileNotFoundError(f"Shared adapter state not found: {model_revision}")
        return load_shared_adapter_state_payload(path)

    def save_manifest(self, payload: ModelManifestPayload) -> Path:
        path = self.manifest_path_for_revision(payload.model_revision)
        dump_model_manifest_payload(path, payload)
        return path

    def load_manifest(self, model_revision: str) -> ModelManifestPayload:
        path = self.manifest_path_for_revision(model_revision)
        if not path.exists():
            raise FileNotFoundError(f"Model manifest not found: {model_revision}")
        return load_model_manifest_payload(path)

    def load_active_pointer(self) -> SharedAdapterStateActivationPointer | None:
        if not self.active_pointer_path.exists():
            return None
        return SharedAdapterStateActivationPointer.model_validate_json(
            self.active_pointer_path.read_text(encoding="utf-8")
        )

    def set_active(
        self,
        model_revision: str,
        *,
        activated_at: datetime | None = None,
    ) -> SharedAdapterStateActivationPointer:
        if not self.path_for_revision(model_revision).exists():
            raise FileNotFoundError(f"Shared adapter state not found: {model_revision}")
        if not self.manifest_path_for_revision(model_revision).exists():
            raise FileNotFoundError(f"Model manifest not found: {model_revision}")
        pointer = SharedAdapterStateActivationPointer(
            model_revision=model_revision,
            activated_at=activated_at or datetime.now(tz=timezone.utc),
        )
        self.active_pointer_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_pointer_path.write_text(
            json.dumps(pointer.model_dump(mode="json"), indent=2, ensure_ascii=True)
            + "\n",
            encoding="utf-8",
        )
        return pointer

    def save_current(
        self,
        *,
        manifest: ModelManifestPayload,
        state: SharedAdapterStatePayload,
        activated_at: datetime | None = None,
    ) -> SharedAdapterStateActivationPointer:
        if manifest.model_id != state.model_id:
            raise ValueError("manifest.model_id must match state.model_id.")
        if manifest.model_revision != state.model_revision:
            raise ValueError("manifest.model_revision must match state.model_revision.")
        if manifest.training_scope != state.training_scope:
            raise ValueError("manifest.training_scope must match state.training_scope.")
        self.save_state(state)
        self.save_manifest(manifest)
        return self.set_active(manifest.model_revision, activated_at=activated_at)

    def load_active_state(self) -> SharedAdapterStatePayload:
        pointer = self.load_active_pointer()
        if pointer is None:
            raise FileNotFoundError("No active shared adapter state is cached.")
        return self.load_state(pointer.model_revision)

    def load_active_manifest(self) -> ModelManifestPayload:
        pointer = self.load_active_pointer()
        if pointer is None:
            raise FileNotFoundError("No active shared adapter state is cached.")
        return self.load_manifest(pointer.model_revision)
