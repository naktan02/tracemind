"""Experiment catalog API tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from main_server.src.api import experiments as experiments_api
from main_server.src.api.main import app
from main_server.src.infrastructure.repositories import (
    experiment_workspace_repository,
)
from main_server.src.services.experiment_workspace.catalog.constants import (
    AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
    FEDERATED_SIMULATION_RUNTIME_PATH,
)
from main_server.src.services.experiment_workspace.catalog.service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiment_workspace.compiler.service import (
    ExperimentCompilerService,
)
from main_server.src.services.experiment_workspace.workspace_service import (
    ExperimentWorkspaceService,
)
from shared.src.contracts.workspace_manifest_contracts import (
    WorkspaceManifestPayload,
    WorkspaceSelectionPayload,
)


def _find_section(catalog, *, track_name: str, section_name: str):
    track = next(track for track in catalog.tracks if track.track_name == track_name)
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
    assert all(
        field.field_name != "algorithm_name" for field in fixmatch.override_fields
    )

    dataset_presets = _find_section(
        payload,
        track_name="central_adaptation",
        section_name="dataset_presets",
    )
    assert dataset_presets.default_slot_name == "dataset_presets"
    ourafla = _find_item(dataset_presets, "ourafla")
    assert ourafla.metadata["readiness"]["central_fixmatch_ready"] is False
    assert ourafla.metadata["sources"]["train"]["kind"] == "huggingface"
    assert ourafla.metadata["asset_paths"]["unlabeled_query_pool_jsonl"] is None
    assert any(
        field.field_name == "train_jsonl" and field.value_kind == "string"
        for field in ourafla.override_fields
    )
    assert all(
        not field.field_name.startswith("sources") for field in ourafla.override_fields
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

    pseudo_label_algorithms = _find_section(
        payload,
        track_name="central_adaptation",
        section_name="pseudo_label_algorithms",
    )
    confidence_only = _find_item(pseudo_label_algorithms, "fixed_confidence_095")
    assert all(
        field.field_name != "algorithm_name"
        for field in confidence_only.override_fields
    )

    generated_ssl_sources = _find_section(
        payload,
        track_name="central_adaptation",
        section_name="query_sources",
    )
    generated_bootstrap_source = _find_item(
        generated_ssl_sources,
        "generated_query_source__bootstrap_teacher_split30_2026_04_14",
    )
    assert generated_bootstrap_source.compiled_selector_name == "dataset_default"
    assert generated_bootstrap_source.default_override_patch == {
        "train_jsonl": (
            "data/processed/lora_bootstrap_classifier_teacher/"
            "bootstrap_teacher_split30_2026_04_14/teacher_seed_train.jsonl"
        ),
        "unlabeled_jsonl": (
            "data/processed/lora_bootstrap_classifier_teacher/"
            "bootstrap_teacher_split30_2026_04_14/teacher_unlabeled_pool.jsonl"
        ),
    }

    initial_checkpoints = _find_section(
        payload,
        track_name="central_adaptation",
        section_name="initial_checkpoints",
    )
    generated_lora_checkpoint = _find_item(
        initial_checkpoints,
        "generated_initial_checkpoint__lora__stage3_supervised_seed4096_2026_04_20",
    )
    assert generated_lora_checkpoint.compiled_selector_name == "required"
    assert (
        generated_lora_checkpoint.default_override_patch["manifest_path"]
        == "data/processed/lora_classifier_heads/"
        "stage3_supervised_seed4096_2026_04_20.manifest.json"
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
        == "scripts/experiments/central_ssl_control/train_lora_fixmatch.py"
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
    lora_fedavg = _find_item(
        federated_aggregations,
        "lora_classifier.fedavg",
        family_name="lora_classifier",
    )
    assert lora_fedavg.core_method_name == "fedavg"
    assert lora_fedavg.metadata["requires_inline_or_materialized_artifacts"] is True

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
        == "methods/adaptation/diagonal_scale/training_backend.py"
    )
    assert diagonal_training.metadata["payload_format"] == "diagonal_scale_update"
    lora_training = _find_item(training_backends, "lora_classifier_trainer")
    assert lora_training.family_name == "lora_classifier"
    assert lora_training.metadata["requires_raw_text"] is True
    assert lora_training.supported_runtime_paths == (FEDERATED_SIMULATION_RUNTIME_PATH,)

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
    assert dp_clip.source_of_truth == "methods/adaptation/privacy_guards/clip_only.py"

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
        section_name="local_update_profiles",
    )
    baseline = _find_item(profiles, "prototype_pseudo_label_v1")
    assert FEDERATED_SIMULATION_RUNTIME_PATH in baseline.supported_runtime_paths
    assert AGENT_LIVE_STORED_EVENT_RUNTIME_PATH in baseline.supported_runtime_paths
    assert baseline.family_name == "diagonal_scale"

    round_profiles = _find_section(
        payload,
        track_name="federated_runtime",
        section_name="round_runtime_profiles",
    )
    round_profile = _find_item(round_profiles, "fedavg_diagonal_scale")
    assert round_profile.metadata["adapter_family_name"] == "diagonal_scale"
    assert round_profile.metadata["aggregation_backend_name"] == "fedavg"

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
                    section_name="query_sources",
                    variant_profile_name="bootstrap_teacher_split30_2026_04_14",
                    family_name="train_source",
                ),
            ),
            global_override_patch={"train_batch_size": 32},
        ),
        service=compiler_service,
    )

    assert (
        plan.script_path
        == "scripts/experiments/central_ssl_control/train_lora_fixmatch.py"
    )
    assert plan.selection_default_groups == (
        "strategy_axes/ssl/consistency_method=fixmatch_usb_v1",
        "strategy_axes/adaptation/peft_adapter=default",
        "track_presets/central_ssl_control/query_source=bootstrap_teacher_split30_2026_04_14",
    )
    assert "query_ssl_method.temperature=0.7" in plan.hydra_overrides
    assert "lora.rank=16" in plan.hydra_overrides
    assert "train_batch_size=32" in plan.hydra_overrides


def test_experiment_compile_api_builds_preview_from_generated_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    catalog_service = ExperimentCatalogService(repo_root=repo_root)
    compiler_service = ExperimentCompilerService(catalog_service=catalog_service)

    plan = experiments_api.compile_experiment_manifest(
        WorkspaceManifestPayload(
            manifest_id="manifest_fixmatch_generated",
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
                WorkspaceSelectionPayload(
                    slot_name="train_source",
                    section_name="query_sources",
                    variant_profile_name=(
                        "generated_query_source__bootstrap_teacher_split30_2026_04_14"
                    ),
                    family_name="train_source",
                ),
                WorkspaceSelectionPayload(
                    slot_name="initial_checkpoint",
                    section_name="initial_checkpoints",
                    variant_profile_name=(
                        "generated_initial_checkpoint__lora__"
                        "stage3_supervised_seed4096_2026_04_20"
                    ),
                    family_name="initial_checkpoint",
                ),
            ),
        ),
        service=compiler_service,
    )

    assert (
        "track_presets/central_ssl_control/query_source=dataset_default"
        in plan.selection_default_groups
    )
    assert (
        "query_source.train_jsonl="
        "data/processed/lora_bootstrap_classifier_teacher/"
        "bootstrap_teacher_split30_2026_04_14/teacher_seed_train.jsonl"
    ) in plan.hydra_overrides
    assert (
        "query_source.unlabeled_jsonl="
        "data/processed/lora_bootstrap_classifier_teacher/"
        "bootstrap_teacher_split30_2026_04_14/teacher_unlabeled_pool.jsonl"
    ) in plan.hydra_overrides
    assert (
        "strategy_axes/adaptation/initial_checkpoint=required"
        in plan.selection_default_groups
    )
    assert (
        "query_adaptation_initial_checkpoint.manifest_path="
        "data/processed/lora_classifier_heads/"
        "stage3_supervised_seed4096_2026_04_20.manifest.json"
    ) in plan.hydra_overrides


def test_experiment_compile_api_rejects_fixmatch_without_unlabeled_pool() -> None:
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
                    slot_name="local_update_profile",
                    section_name="local_update_profiles",
                    core_method_name="prototype_pseudo_label_v1",
                    variant_profile_name="prototype_pseudo_label_v1",
                    family_name="diagonal_scale",
                ),
            ),
            global_override_patch={"training_task.local_epochs": 2},
        ),
        service=compiler_service,
    )

    assert plan.script_path == "scripts/experiments/fl_ssl/run_federated_simulation.py"
    assert plan.selection_default_groups == (
        "strategy_axes/fl/local_update_profile=prototype_pseudo_label_v1",
    )
    assert plan.hydra_overrides == ("training_task.local_epochs=2",)
    assert any("simulation participant" in warning for warning in plan.warnings)


def test_experiment_compile_api_warns_on_large_label_dominant_client_count() -> None:
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


def test_saved_workspace_api_lists_selection_previews_and_supports_delete(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    service = ExperimentWorkspaceService(
        compiler_service=ExperimentCompilerService(
            catalog_service=ExperimentCatalogService(repo_root=repo_root)
        ),
        workspace_repository=experiment_workspace_repository.ExperimentWorkspaceRepository(
            experiments_root=tmp_path / "experiments"
        ),
    )

    saved = experiments_api.save_experiment_workspace(
        WorkspaceManifestPayload(
            manifest_id="manifest_seed_compare",
            track_name="seed",
            entrypoint_name="train_softmax_classifier",
            selections=(
                WorkspaceSelectionPayload(
                    slot_name="dataset_presets",
                    section_name="dataset_presets",
                    variant_profile_name="ourafla",
                    family_name="dataset",
                    override_patch={"train_jsonl": "custom.jsonl"},
                ),
            ),
        ),
        service=service,
    )

    listed = experiments_api.list_saved_experiment_workspaces(service=service)
    assert listed[0].workspace_id == saved.workspace_id
    assert listed[0].selection_previews[0].section_name == "dataset_presets"
    assert listed[0].selection_previews[0].override_keys == ("train_jsonl",)

    deleted = experiments_api.delete_saved_experiment_workspace(
        saved.workspace_id,
        service=service,
    )
    assert deleted.workspace_id == saved.workspace_id
    assert experiments_api.list_saved_experiment_workspaces(service=service) == ()


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
