"""experiment_web generated type source bridge."""

from __future__ import annotations

from typing import Any

import shared.src.contracts.workspace_manifest_contracts as workspace_manifest_contracts


def build_experiment_web_type_sources() -> tuple[tuple[str, Any], ...]:
    """experiment_web가 노출받는 payload/schema source를 반환한다."""

    import main_server.src.services.experiment_workspace.payloads as experiment_payloads

    return (
        ("CatalogItemCompileSupport", experiment_payloads.CatalogItemCompileSupport),
        (
            "CatalogOverrideFieldValueKind",
            experiment_payloads.CatalogOverrideFieldValueKind,
        ),
        (
            "CatalogSectionSelectionMode",
            experiment_payloads.CatalogSectionSelectionMode,
        ),
        (
            "WorkspaceConfigScalar",
            workspace_manifest_contracts.WorkspaceConfigScalar,
        ),
        (
            "CatalogOverrideFieldPayload",
            experiment_payloads.CatalogOverrideFieldPayload,
        ),
        ("CatalogItemPayload", experiment_payloads.CatalogItemPayload),
        ("CatalogSectionPayload", experiment_payloads.CatalogSectionPayload),
        ("CatalogTrackPayload", experiment_payloads.CatalogTrackPayload),
        ("ExperimentCatalogPayload", experiment_payloads.ExperimentCatalogPayload),
        (
            "SavedWorkspaceSummaryPayload",
            experiment_payloads.SavedWorkspaceSummaryPayload,
        ),
        (
            "SavedWorkspaceSelectionPreviewPayload",
            experiment_payloads.SavedWorkspaceSelectionPreviewPayload,
        ),
        (
            "SavedWorkspaceDetailPayload",
            experiment_payloads.SavedWorkspaceDetailPayload,
        ),
        (
            "LaunchExperimentRunRequestPayload",
            experiment_payloads.LaunchExperimentRunRequestPayload,
        ),
        ("ExperimentRunMetricPayload", experiment_payloads.ExperimentRunMetricPayload),
        (
            "ExperimentRunResultSummaryPayload",
            experiment_payloads.ExperimentRunResultSummaryPayload,
        ),
        ("ExperimentRunStatus", experiment_payloads.ExperimentRunStatus),
        ("ExperimentRunPayload", experiment_payloads.ExperimentRunPayload),
        (
            "WorkspaceSelectionPayload",
            workspace_manifest_contracts.WorkspaceSelectionPayload,
        ),
        (
            "WorkspaceManifestPayload",
            workspace_manifest_contracts.WorkspaceManifestPayload,
        ),
        (
            "ResolvedWorkspaceSelectionPayload",
            workspace_manifest_contracts.ResolvedWorkspaceSelectionPayload,
        ),
        (
            "ResolvedExperimentPlanPayload",
            workspace_manifest_contracts.ResolvedExperimentPlanPayload,
        ),
    )
