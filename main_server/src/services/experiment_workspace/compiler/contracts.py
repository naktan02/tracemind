"""Experiment compile policy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from main_server.src.services.experiment_workspace.catalog.service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiment_workspace.payloads import CatalogItemPayload
from shared.src.contracts.workspace_manifest_contracts import (
    WorkspaceManifestPayload,
)


@dataclass(frozen=True, slots=True)
class ExperimentCompileContext:
    """Entrypoint-specific compile policy가 읽는 canonical context."""

    manifest: WorkspaceManifestPayload
    entrypoint_item: CatalogItemPayload
    effective_groups: dict[str, str]
    hydra_override_map: dict[str, str]
    catalog_service: ExperimentCatalogService


class ExperimentCompilePolicy(Protocol):
    """Entrypoint-specific compile policy."""

    def collect_warnings(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> tuple[str, ...]:
        """compile preview에 추가할 warning을 반환한다."""

    def validate_requirements(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> None:
        """compile 전제조건을 검사하고 실패 시 ValueError를 올린다."""


@dataclass(frozen=True, slots=True)
class NoOpExperimentCompilePolicy:
    """추가 warning/validation이 없는 기본 policy."""

    def collect_warnings(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> tuple[str, ...]:
        del context
        return ()

    def validate_requirements(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> None:
        del context
