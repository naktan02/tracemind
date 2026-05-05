from __future__ import annotations

from pathlib import Path

from shared.src.contracts.workspace_manifest_contracts import (
    ResolvedExperimentPlanPayload,
    ResolvedWorkspaceSelectionPayload,
    WorkspaceManifestPayload,
    WorkspaceSelectionPayload,
    dump_resolved_experiment_plan_payload,
    dump_workspace_manifest_payload,
    load_resolved_experiment_plan_payload,
    load_workspace_manifest_payload,
)


def test_workspace_manifest_payload_accepts_core_variant_and_override_patch() -> None:
    payload = WorkspaceManifestPayload(
        manifest_id="manifest_001",
        track_name="central_adaptation",
        entrypoint_name="train_lora_fixmatch",
        selections=(
            WorkspaceSelectionPayload(
                slot_name="ssl_method",
                section_name="query_ssl_methods",
                core_method_name="fixmatch",
                variant_profile_name="fixmatch_usb_v1",
                family_name="ssl_method",
                override_patch={"temperature": 0.7, "require_multiview": True},
            ),
        ),
        global_override_patch={"train_batch_size": 32},
    )

    selection = payload.selections[0]
    assert selection.core_method_name == "fixmatch"
    assert selection.variant_profile_name == "fixmatch_usb_v1"
    assert selection.override_patch["temperature"] == 0.7
    assert payload.global_override_patch["train_batch_size"] == 32


def test_workspace_manifest_payload_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "workspace_manifest.json"
    payload = WorkspaceManifestPayload(
        manifest_id="manifest_001",
        track_name="federated_runtime",
        entrypoint_name="run_federated_simulation",
        selections=(
            WorkspaceSelectionPayload(
                slot_name="training_algorithm_profile",
                section_name="training_algorithm_profiles",
                variant_profile_name="prototype_pseudo_label_v1",
                family_name="diagonal_scale",
            ),
        ),
        global_override_patch={"training_task.local_epochs": 2},
    )

    dump_workspace_manifest_payload(path, payload)
    loaded = load_workspace_manifest_payload(path)

    assert loaded == payload


def test_resolved_experiment_plan_payload_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "resolved_plan.json"
    payload = ResolvedExperimentPlanPayload(
        manifest_id="manifest_001",
        track_name="central_adaptation",
        entrypoint_name="train_lora_fixmatch",
        job_config_path="conf/entrypoints/central_ssl_control/train_lora_fixmatch.yaml",
        script_path="scripts/experiments/central_ssl_control/train_lora_fixmatch.py",
        base_default_groups=(
            "execution_context/dataset_asset=ourafla",
            "execution_context/runtime_env=gpu_online",
        ),
        selection_default_groups=(
            "strategy_axes/ssl/consistency_method=fixmatch_usb_v1",
        ),
        hydra_overrides=("query_ssl_method.temperature=0.7",),
        command_args=(
            "uv",
            "run",
            "python",
            "scripts/experiments/central_ssl_control/train_lora_fixmatch.py",
            "strategy_axes/ssl/consistency_method=fixmatch_usb_v1",
            "query_ssl_method.temperature=0.7",
        ),
        resolved_selections=(
            ResolvedWorkspaceSelectionPayload(
                slot_name="ssl_method",
                section_name="query_ssl_methods",
                family_name="ssl_method",
                core_method_name="fixmatch",
                variant_profile_name="fixmatch_usb_v1",
                source_of_truth=(
                    "conf/strategy_axes/ssl/consistency_method/fixmatch_usb_v1.yaml"
                ),
                preset_group="query_ssl_method",
                compiled_selector="strategy_axes/ssl/consistency_method=fixmatch_usb_v1",
                compiled_overrides=("query_ssl_method.temperature=0.7",),
            ),
        ),
    )

    dump_resolved_experiment_plan_payload(path, payload)
    loaded = load_resolved_experiment_plan_payload(path)

    assert loaded == payload
