"""Compatibility facade for federated simulation server runtime helpers."""
# ruff: noqa: F401

from __future__ import annotations

from scripts.runtime_adapters.federated_server.initial_state_factory import (
    build_classifier_head_state_from_prototype_pack,
    build_initial_shared_state,
)
from scripts.runtime_adapters.federated_server.prototype_rebuild_bridge import (
    SimulationEmbeddingAdapterFactory,
    build_prototype_rebuild_runtime_service,
    rebuild_reference_prototype_pack,
    store_prototype_rebuild_input,
)
from scripts.runtime_adapters.federated_server.round_request_mapper import (
    build_federated_training_task_config,
    build_round_open_request,
)
from scripts.runtime_adapters.federated_server.runtime import (
    SimulationServerRuntime,
    build_simulation_round_family,
    load_active_state,
)
