"""Experiment catalog API tests."""

from __future__ import annotations

from pathlib import Path

from main_server.src.api import experiments as experiments_api
from main_server.src.api.main import app
from main_server.src.services.experiments.catalog_service import (
    AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
    FEDERATED_SIMULATION_RUNTIME_PATH,
    ExperimentCatalogService,
)
from main_server.src.services.experiments.compiler_service import (
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


def _find_item(section, item_name: str):
    return next(item for item in section.items if item.item_name == item_name)


def test_experiment_catalog_api_lists_current_strategy_inventory() -> None:
    service = ExperimentCatalogService(repo_root=Path(__file__).resolve().parents[3])

    payload = experiments_api.get_experiment_catalog(service=service)

    assert [track.track_name for track in payload.tracks] == [
        "seed",
        "central_adaptation",
        "federated_runtime",
    ]

    central_ssl_methods = _find_section(
        payload,
        track_name="central_adaptation",
        section_name="query_ssl_methods",
    )
    fixmatch = _find_item(central_ssl_methods, "fixmatch_usb_v1")
    assert fixmatch.core_method_name == "fixmatch"
    assert fixmatch.variant_profile_name == "fixmatch_usb_v1"
    assert fixmatch.metadata["require_multiview"] is True

    federated_aggregations = _find_section(
        payload,
        track_name="federated_runtime",
        section_name="aggregation_backends",
    )
    fedavg = _find_item(federated_aggregations, "fedavg")
    assert fedavg.family_name in {"classifier_head", "diagonal_scale"}


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


def test_experiments_router_is_registered_on_main_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/experiments/catalog" in route_paths
    assert "/api/v1/experiments/compile" in route_paths


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
            ),
            global_override_patch={"train_batch_size": 32},
        ),
        service=compiler_service,
    )

    assert plan.script_path == "scripts/experiments/train_lora_fixmatch.py"
    assert plan.selection_default_groups == (
        "query_ssl_method=fixmatch_usb_v1",
        "lora=default",
    )
    assert "query_ssl_method.temperature=0.7" in plan.hydra_overrides
    assert "lora.rank=16" in plan.hydra_overrides
    assert "train_batch_size=32" in plan.hydra_overrides


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
