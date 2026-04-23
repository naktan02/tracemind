"""Experiment catalog API tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from main_server.src.api import experiments as experiments_api
from main_server.src.api.main import app
from main_server.src.services.experiment_workspace.catalog_constants import (
    AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
    FEDERATED_SIMULATION_RUNTIME_PATH,
)
from main_server.src.services.experiment_workspace.catalog_service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiment_workspace.compiler_service import (
    ExperimentCompilerService,
)
from shared.src.contracts.workspace_manifest_contracts import (
    WorkspaceManifestPayload,
    WorkspaceSelectionPayload,
)


def _find_section(catalog, *, track_name: str, section_name: str):
    track = next(
        track for track in catalog.tracks if track.track_name == track_name
    )
    return next(
        section for section in track.sections if section.section_name == section_name
    )


def _find_item(section, item_name: str, *, family_name: str | None = None):
    return next(
        item
        for item in section.items
        if item.item_name == item_name
        and (family_name is None or item.family_name == family_name)
    )


def test_experiment_catalog_api_lists_current_strategy_inventory() -> None:
    service = ExperimentCatalogService(repo_root=Path(__file__).resolve().parents[3])

    payload = experiments_api.get_experiment_catalog(service=service)

    assert [track.track_name for track in payload.tracks] == [
        "seed",
        "central_adaptation",
        "federated_runtime",
    ]
    assert payload.tracks[1].entrypoint_section_name == "entrypoints"

    central_ssl_methods = _find_section(
        payload,
        track_name="central_adaptation",
        section_name="query_ssl_methods",
    )
    fixmatch = _find_item(central_ssl_methods, "fixmatch_usb_v1")
    assert fixmatch.core_method_name == "fixmatch"
    assert fixmatch.variant_profile_name == "fixmatch_usb_v1"
    assert fixmatch.compile_support == "preset_selector"
    assert fixmatch.metadata["require_multiview"] is True

    dataset_presets = _find_section(
        payload,
        track_name="central_adaptation",
        section_name="dataset_presets",
    )
    assert dataset_presets.default_slot_name == "dataset_presets"
    ourafla = _find_item(dataset_presets, "ourafla")
    assert ourafla.metadata["readiness"]["central_fixmatch_ready"] is False
    assert ourafla.metadata["sources"]["train"]["kind"] == "huggingface"
    assert (
        ourafla.metadata["asset_paths"]["unlabeled_query_pool_jsonl"] is None
    )
    assert any(
        field.field_name == "train_jsonl" and field.value_kind == "string"
        for field in ourafla.override_fields
    )
    assert all(
        not field.field_name.startswith("sources")
        for field in ourafla.override_fields
    )

    peft_methods = _find_section(
        payload,
        track_name="central_adaptation",
        section_name="peft_methods",
    )
    default_lora = _find_item(peft_methods, "default")
    assert any(
        field.field_name == "rank"
        and field.value_kind == "integer"
        and field.default_value == 8
        for field in default_lora.override_fields
    )
    assert any(
        field.field_name == "use_rslora"
        and field.value_kind == "boolean"
        and field.default_value is False
        for field in default_lora.override_fields
    )

    central_entrypoints = _find_section(
        payload,
        track_name="central_adaptation",
        section_name="entrypoints",
    )
    assert central_entrypoints.selection_mode == "single_required"
    fixmatch_entrypoint = _find_item(central_entrypoints, "train_lora_fixmatch")
    assert fixmatch_entrypoint.compile_support == "entrypoint"
    assert (
        fixmatch_entrypoint.script_path
        == "scripts/experiments/train_lora_fixmatch.py"
    )

    federated_aggregations = _find_section(
        payload,
        track_name="federated_runtime",
        section_name="aggregation_backends",
    )
    fedavg = _find_item(
        federated_aggregations,
        "classifier_head.fedavg",
        family_name="classifier_head",
    )
    assert fedavg.core_method_name == "fedavg"
    assert fedavg.compile_support == "metadata_only"
    assert fedavg.compile_blocker_reason is not None

    training_backends = _find_section(
        payload,
        track_name="federated_runtime",
        section_name="training_backends",
    )
    diagonal_training = _find_item(
        training_backends,
        "diagonal_scale_heuristic",
    )
    assert (
        diagonal_training.source_of_truth
        == "agent/src/services/training/backends/training/diagonal_scale_heuristic.py"
    )
    assert diagonal_training.metadata["payload_format"] == "diagonal_scale_update"

    example_backends = _find_section(
        payload,
        track_name="federated_runtime",
        section_name="example_generation_backends",
    )
    weak_strong_example = _find_item(example_backends, "weak_strong_pair")
    assert (
        weak_strong_example.source_of_truth
        == "agent/src/services/training/backends/inputs/weak_strong_pair.py"
    )

    privacy_guards = _find_section(
        payload,
        track_name="federated_runtime",
        section_name="privacy_guards",
    )
    dp_clip = _find_item(privacy_guards, "classifier_head_clip_only")
    assert (
        dp_clip.source_of_truth
        == "agent/src/services/training/execution/privacy_guard_service.py"
    )

    federated_presets = _find_section(
        payload,
        track_name="federated_runtime",
        section_name="federated_run_presets",
    )
    smoke = _find_item(federated_presets, "smoke")
    assert smoke.metadata["count_semantics"]["client_count"] == (
        "simulation_participants"
    )


def test_experiment_catalog_api_exposes_runtime_compatibility_metadata() -> None:
    service = ExperimentCatalogService(repo_root=Path(__file__).resolve().parents[3])

    payload = experiments_api.get_experiment_catalog(service=service)

    profiles = _find_section(
        payload,
        track_name="federated_runtime",
        section_name="training_algorithm_profiles",
    )
    baseline = _find_item(profiles, "prototype_pseudo_label_v1")
    assert FEDERATED_SIMULATION_RUNTIME_PATH in baseline.supported_runtime_paths
    assert AGENT_LIVE_STORED_EVENT_RUNTIME_PATH in baseline.supported_runtime_paths

    example_backends = _find_section(
        payload,
        track_name="federated_runtime",
        section_name="example_generation_backends",
    )
    weak_strong = _find_item(example_backends, "weak_strong_pair")
    assert weak_strong.supported_runtime_paths == (FEDERATED_SIMULATION_RUNTIME_PATH,)

    scoring_backends = _find_section(
        payload,
        track_name="federated_runtime",
        section_name="scoring_backends",
    )
    prototype_similarity = _find_item(
        scoring_backends,
        "prototype_similarity",
    )
    assert prototype_similarity.metadata["confidence_kind"] == (
        "prototype_similarity_top1"
    )


def test_experiments_router_is_registered_on_main_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/experiments/catalog" in route_paths
    assert "/api/v1/experiments/compile" in route_paths
    assert "/api/v1/experiments/workspaces" in route_paths
    assert "/api/v1/experiments/workspaces/{workspace_id}" in route_paths
    assert "/api/v1/experiments/runs" in route_paths
    assert "/api/v1/experiments/runs/{run_id}" in route_paths
    assert "/api/v1/experiments/runs/{run_id}/logs/{stream_name}" in route_paths


def test_experiment_compile_api_builds_central_adaptation_preview() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    catalog_service = ExperimentCatalogService(repo_root=repo_root)
    compiler_service = ExperimentCompilerService(catalog_service=catalog_service)

    plan = experiments_api.compile_experiment_manifest(
        WorkspaceManifestPayload(
            manifest_id="manifest_fixmatch",
            track_name="central_adaptation",
            entrypoint_name="train_lora_fixmatch",
            selections=(
                WorkspaceSelectionPayload(
                    slot_name="ssl_method",
                    section_name="query_ssl_methods",
                    core_method_name="fixmatch",
                    variant_profile_name="fixmatch_usb_v1",
                    family_name="ssl_method",
                    override_patch={"temperature": 0.7},
                ),
                WorkspaceSelectionPayload(
                    slot_name="peft_method",
                    section_name="peft_methods",
                    core_method_name="lora",
                    variant_profile_name="default",
                    family_name="peft_adapter",
                    override_patch={"rank": 16},
                ),
                WorkspaceSelectionPayload(
                    slot_name="train_source",
                    section_name="query_ssl_train_sources",
                    variant_profile_name="bootstrap_teacher_split30_2026_04_14",
                    family_name="train_source",
                ),
            ),
            global_override_patch={"train_batch_size": 32},
        ),
        service=compiler_service,
    )

    assert plan.script_path == "scripts/experiments/train_lora_fixmatch.py"
    assert plan.selection_default_groups == (
        "query_ssl_method=fixmatch_usb_v1",
        "lora=default",
        "query_ssl_train_source=bootstrap_teacher_split30_2026_04_14",
    )
    assert "query_ssl_method.temperature=0.7" in plan.hydra_overrides
    assert "lora.rank=16" in plan.hydra_overrides
    assert "train_batch_size=32" in plan.hydra_overrides


def test_experiment_compile_api_rejects_fixmatch_when_dataset_lacks_unlabeled_pool(
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    catalog_service = ExperimentCatalogService(repo_root=repo_root)
    compiler_service = ExperimentCompilerService(catalog_service=catalog_service)

    with pytest.raises(HTTPException) as error:
        experiments_api.compile_experiment_manifest(
            WorkspaceManifestPayload(
                manifest_id="manifest_fixmatch_missing_unlabeled",
                track_name="central_adaptation",
                entrypoint_name="train_lora_fixmatch",
                selections=(
                    WorkspaceSelectionPayload(
                        slot_name="ssl_method",
                        section_name="query_ssl_methods",
                        core_method_name="fixmatch",
                        variant_profile_name="fixmatch_usb_v1",
                        family_name="ssl_method",
                    ),
                ),
            ),
            service=compiler_service,
        )

    assert error.value.status_code == 400
    assert "unlabeled_query_pool_jsonl" in error.value.detail


def test_experiment_compile_api_builds_federated_preview() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    catalog_service = ExperimentCatalogService(repo_root=repo_root)
    compiler_service = ExperimentCompilerService(catalog_service=catalog_service)

    plan = experiments_api.compile_experiment_manifest(
        WorkspaceManifestPayload(
            manifest_id="manifest_federated",
            track_name="federated_runtime",
            entrypoint_name="run_federated_simulation",
            selections=(
                WorkspaceSelectionPayload(
                    slot_name="training_algorithm_profile",
                    section_name="training_algorithm_profiles",
                    core_method_name="prototype_pseudo_label_v1",
                    variant_profile_name="prototype_pseudo_label_v1",
                    family_name="diagonal_scale",
                ),
            ),
            global_override_patch={"training_task.local_epochs": 2},
        ),
        service=compiler_service,
    )

    assert plan.script_path == "scripts/experiments/run_federated_simulation.py"
    assert plan.selection_default_groups == (
        "training_algorithm_profile=prototype_pseudo_label_v1",
    )
    assert plan.hydra_overrides == ("training_task.local_epochs=2",)
    assert any("simulation participant" in warning for warning in plan.warnings)


def test_experiment_compile_api_warns_when_client_count_outgrows_label_dominant_shards(
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    catalog_service = ExperimentCatalogService(repo_root=repo_root)
    compiler_service = ExperimentCompilerService(catalog_service=catalog_service)

    plan = experiments_api.compile_experiment_manifest(
        WorkspaceManifestPayload(
            manifest_id="manifest_federated_large_client_count",
            track_name="federated_runtime",
            entrypoint_name="run_federated_simulation",
            global_override_patch={"federated_run_preset.client_count": 6},
        ),
        service=compiler_service,
    )

    assert any("빈 shard" in warning for warning in plan.warnings)


def test_experiment_compile_api_rejects_metadata_only_selection() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    catalog_service = ExperimentCatalogService(repo_root=repo_root)
    compiler_service = ExperimentCompilerService(catalog_service=catalog_service)

    with pytest.raises(HTTPException) as error:
        experiments_api.compile_experiment_manifest(
            WorkspaceManifestPayload(
                manifest_id="manifest_invalid_aggregation",
                track_name="federated_runtime",
                entrypoint_name="run_federated_simulation",
                selections=(
                    WorkspaceSelectionPayload(
                        slot_name="aggregation_backend",
                        section_name="aggregation_backends",
                        core_method_name="fedavg",
                        variant_profile_name="classifier_head.fedavg",
                        family_name="classifier_head",
                    ),
                ),
            ),
            service=compiler_service,
        )

    assert error.value.status_code == 400
    assert "not compileable yet" in error.value.detail
