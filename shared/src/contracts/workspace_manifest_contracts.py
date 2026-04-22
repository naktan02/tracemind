"""개발자용 experiment workspace manifest/compile contract."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

WorkspaceConfigScalar = str | int | float | bool | None


class WorkspaceSelectionPayload(BaseModel):
    """Workspace 안의 개별 block selection.

    `core_method_name`은 재사용 가능한 구현 코어를,
    `variant_profile_name`은 그 위의 named preset을,
    `override_patch`는 해당 variant 위에 덧씌우는 run-local override를 뜻한다.
    """

    model_config = ConfigDict(extra="forbid")

    slot_name: str = Field(
        description="UI/workspace 안에서 이 selection이 놓이는 slot."
    )
    section_name: str = Field(
        description="Catalog track 안에서 이 selection을 찾을 section 이름."
    )
    variant_profile_name: str = Field(
        description="선택한 variant profile 또는 named preset 식별자."
    )
    core_method_name: str | None = Field(
        default=None,
        description="고정된 implementation core 식별자. 없으면 profile-only selection.",
    )
    family_name: str | None = Field(
        default=None,
        description="선택이 속해야 하는 component/strategy family.",
    )
    override_patch: dict[str, WorkspaceConfigScalar] = Field(
        default_factory=dict,
        description="variant profile 위에 덧씌우는 run-local override patch.",
    )


class WorkspaceManifestPayload(BaseModel):
    """웹/CLI가 편집하는 canonical experiment workspace 문서."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "workspace_manifest.v1"
    manifest_id: str = Field(description="Workspace manifest 고유 식별자.")
    track_name: str = Field(
        description="seed/central_adaptation/federated_runtime track."
    )
    entrypoint_name: str = Field(description="compile 대상 experiment entrypoint 이름.")
    selections: tuple[WorkspaceSelectionPayload, ...] = Field(
        default_factory=tuple,
        description="entrypoint 위에 덧붙이는 block selection 목록.",
    )
    global_override_patch: dict[str, WorkspaceConfigScalar] = Field(
        default_factory=dict,
        description="개별 selection에 속하지 않는 top-level override patch.",
    )
    notes: str | None = Field(
        default=None,
        description="개발자 메모 또는 실험 설명.",
    )


class ResolvedWorkspaceSelectionPayload(BaseModel):
    """Catalog와 compiler가 검증을 끝낸 selection 요약."""

    model_config = ConfigDict(extra="forbid")

    slot_name: str
    section_name: str
    family_name: str | None = None
    core_method_name: str | None = None
    variant_profile_name: str
    source_of_truth: str
    preset_group: str | None = None
    compiled_selector: str | None = None
    compiled_overrides: tuple[str, ...] = Field(default_factory=tuple)


class ResolvedExperimentPlanPayload(BaseModel):
    """WorkspaceManifest를 기존 Hydra/script 실행 표면으로 번역한 결과."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "resolved_experiment_plan.v1"
    manifest_id: str
    track_name: str
    entrypoint_name: str
    job_config_path: str
    script_path: str
    base_default_groups: tuple[str, ...] = Field(default_factory=tuple)
    selection_default_groups: tuple[str, ...] = Field(default_factory=tuple)
    hydra_overrides: tuple[str, ...] = Field(default_factory=tuple)
    command_args: tuple[str, ...] = Field(default_factory=tuple)
    resolved_selections: tuple[ResolvedWorkspaceSelectionPayload, ...] = Field(
        default_factory=tuple
    )
    warnings: tuple[str, ...] = Field(default_factory=tuple)


def _dump_payload(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def dump_workspace_manifest_payload(
    path: Path,
    payload: WorkspaceManifestPayload,
) -> None:
    """Workspace manifest를 JSON 파일로 기록한다."""

    _dump_payload(path, payload)


def load_workspace_manifest_payload(path: Path) -> WorkspaceManifestPayload:
    """JSON 파일에서 workspace manifest를 읽는다."""

    return WorkspaceManifestPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_resolved_experiment_plan_payload(
    path: Path,
    payload: ResolvedExperimentPlanPayload,
) -> None:
    """Resolved experiment plan을 JSON 파일로 기록한다."""

    _dump_payload(path, payload)


def load_resolved_experiment_plan_payload(
    path: Path,
) -> ResolvedExperimentPlanPayload:
    """JSON 파일에서 resolved experiment plan을 읽는다."""

    return ResolvedExperimentPlanPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


WorkspaceManifest = WorkspaceManifestPayload
ResolvedExperimentPlan = ResolvedExperimentPlanPayload


__all__ = [
    "ResolvedExperimentPlan",
    "ResolvedExperimentPlanPayload",
    "ResolvedWorkspaceSelectionPayload",
    "WorkspaceConfigScalar",
    "WorkspaceManifest",
    "WorkspaceManifestPayload",
    "WorkspaceSelectionPayload",
    "dump_resolved_experiment_plan_payload",
    "dump_workspace_manifest_payload",
    "load_resolved_experiment_plan_payload",
    "load_workspace_manifest_payload",
]
