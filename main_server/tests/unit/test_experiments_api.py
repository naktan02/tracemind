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
    assert fixmatch.method_name == "fixmatch"
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
