"""Repository wiring for federated simulation server runtime."""

from __future__ import annotations

from pathlib import Path

from main_server.src.infrastructure.repositories import (
    model_manifest_repository as model_manifest_repo,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as adapter_state_repo,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_update_repository as adapter_update_repo,
)
from main_server.src.infrastructure.repositories.round_repository import RoundRepository


def build_shared_adapter_state_repository(
    output_dir: Path,
) -> adapter_state_repo.SharedAdapterStateRepository:
    return adapter_state_repo.SharedAdapterStateRepository(
        state_root=output_dir / "main_server" / "shared_adapter_states"
    )


def build_round_repository(output_dir: Path) -> RoundRepository:
    return RoundRepository(state_root=output_dir / "main_server" / "rounds")


def build_shared_adapter_update_repository(
    output_dir: Path,
) -> adapter_update_repo.SharedAdapterUpdateRepository:
    return adapter_update_repo.SharedAdapterUpdateRepository(
        state_root=output_dir / "main_server" / "shared_adapter_updates"
    )


def build_model_manifest_repository(
    output_dir: Path,
) -> model_manifest_repo.ModelManifestRepository:
    return model_manifest_repo.ModelManifestRepository(
        state_root=output_dir / "main_server" / "model_manifests"
    )
