"""Repository wiring for federated simulation server runtime."""

from __future__ import annotations

from pathlib import Path

from main_server.src.infrastructure.repositories import (
    model_manifest_repository as model_manifest_repo,
)
from main_server.src.infrastructure.repositories import (
    prototype_build_state_repository as build_state_repo,
)
from main_server.src.infrastructure.repositories import (
    prototype_pack_repository as pack_repo,
)
from main_server.src.infrastructure.repositories import (
    prototype_rebuild_input_repository as rebuild_input_repo,
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


def build_prototype_rebuild_input_repository(
    output_dir: Path,
) -> rebuild_input_repo.PrototypeRebuildInputRepository:
    return rebuild_input_repo.PrototypeRebuildInputRepository(
        state_root=output_dir / "main_server" / "prototype_rebuild_inputs"
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


def build_runtime_prototype_pack_repository(
    output_dir: Path,
) -> pack_repo.PrototypePackRepository:
    return pack_repo.PrototypePackRepository(
        state_root=output_dir / "main_server" / "runtime_prototype_packs"
    )


def build_runtime_prototype_build_state_repository(
    output_dir: Path,
) -> build_state_repo.PrototypeBuildStateRepository:
    return build_state_repo.PrototypeBuildStateRepository(
        state_root=output_dir / "main_server" / "runtime_prototype_build_states"
    )
