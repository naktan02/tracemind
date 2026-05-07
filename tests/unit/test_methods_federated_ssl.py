"""Reusable FL SSL method descriptor tests."""

from __future__ import annotations

import pytest

from methods.common.registry import MethodRegistry
from methods.federated_ssl import registry as federated_ssl_registry
from methods.federated_ssl.base import (
    FederatedSslLocalStepSpec,
    FederatedSslMethodDescriptor,
    FederatedSslRequiredViews,
    FederatedSslRuntimeCapabilities,
    FederatedSslServerStepSpec,
)
from methods.federated_ssl.compatibility import (
    FederatedSslProfileCompatibilityContext,
    validate_federated_ssl_profile_compatibility,
)
from methods.federated_ssl.fedavg_pseudo_label.descriptor import (
    FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
)
from methods.federated_ssl.fedavg_pseudo_label.fedavg_pseudo_label import (
    descriptor as fedavg_pseudo_label_descriptor,
)
from methods.federated_ssl.fedavg_pseudo_label.fedavg_pseudo_label import (
    local_objective as fedavg_pseudo_label_local_objective,
)
from methods.federated_ssl.fedavg_pseudo_label.fedavg_pseudo_label import (
    round_policy as fedavg_pseudo_label_round_policy,
)
from methods.federated_ssl.fedavg_pseudo_label.fedavg_pseudo_label import (
    server_policy as fedavg_pseudo_label_server_policy,
)
from methods.federated_ssl.local_update_profile import (
    LocalUpdateProfile,
    require_training_objective_matches_local_update_profile,
)
from methods.federated_ssl.registry import (
    list_federated_ssl_method_descriptors,
    resolve_federated_ssl_method_descriptor,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


def test_federated_ssl_descriptor_registry_resolves_active_baseline() -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedavg_pseudo_label")

    assert descriptor is FEDAVG_PSEUDO_LABEL_DESCRIPTOR
    assert descriptor.implementation_status == "active_runtime"
    assert descriptor.required_views.view_names == ("single_view",)
    assert descriptor.client_trainer_name == "local_training_service"
    assert descriptor.pseudo_labeler_name == "ssl_pseudo_label_selection_hook"
    assert descriptor.view_generator_name == "training_example_backend"
    assert descriptor.server_aggregator_name == "round_runtime_aggregation_backend"
    assert descriptor.server_step.server_aggregate_hint == (
        "use_round_runtime_aggregation_backend"
    )
    assert descriptor.requires_custom_client_runtime is False
    assert descriptor.requires_custom_server_runtime is False
    assert descriptor.runtime_capabilities.simulation_supported is True
    assert descriptor.runtime_capabilities.live_agent_supported is True
    assert descriptor.runtime_capabilities.live_server_supported is True


def test_fedavg_pseudo_label_exposes_method_local_policy_seams() -> None:
    assert fedavg_pseudo_label_descriptor is FEDAVG_PSEUDO_LABEL_DESCRIPTOR
    assert fedavg_pseudo_label_local_objective.objective_name == (
        FEDAVG_PSEUDO_LABEL_DESCRIPTOR.local_step.step_name
    )
    assert fedavg_pseudo_label_server_policy.policy_name == (
        FEDAVG_PSEUDO_LABEL_DESCRIPTOR.server_step.server_aggregator_name
    )
    assert fedavg_pseudo_label_round_policy.policy_name == (
        FEDAVG_PSEUDO_LABEL_DESCRIPTOR.server_step.round_policy_name
    )
    assert fedavg_pseudo_label_round_policy.custom_round_policy_required is False


def test_federated_ssl_descriptor_registry_rejects_unwired_method() -> None:
    with pytest.raises(NotImplementedError, match="descriptor is not wired yet"):
        resolve_federated_ssl_method_descriptor("paper_method_candidate")


def test_federated_ssl_descriptor_registry_lists_unique_descriptors() -> None:
    descriptors = list_federated_ssl_method_descriptors(
        method_names=("fedavg_pseudo_label", "fedavg_pseudo_label")
    )

    assert descriptors == (FEDAVG_PSEUDO_LABEL_DESCRIPTOR,)


def test_federated_ssl_registry_supports_test_only_method_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_registry = MethodRegistry[FederatedSslMethodDescriptor](
        item_label="test federated SSL method descriptor"
    )
    monkeypatch.setattr(
        federated_ssl_registry,
        "_FEDERATED_SSL_METHOD_DESCRIPTORS",
        isolated_registry,
    )
    monkeypatch.setattr(
        federated_ssl_registry,
        "_BUILTIN_FEDERATED_SSL_METHODS_LOADED",
        True,
    )

    dummy_descriptor = FederatedSslMethodDescriptor(
        name="dummy_federated_ssl_method",
        implementation_status="test_only",
        required_views=FederatedSslRequiredViews(
            view_names=("single_view",),
            view_generator_name="test_view_generator",
        ),
        local_step=FederatedSslLocalStepSpec(
            step_name="dummy_local_step",
            client_trainer_name="dummy_client_trainer",
            pseudo_labeler_name="dummy_pseudo_labeler",
        ),
        server_step=FederatedSslServerStepSpec(
            server_aggregator_name="dummy_aggregator",
            round_policy_name="dummy_round_policy",
            server_aggregate_hint="dummy_aggregate_hint",
        ),
        runtime_capabilities=FederatedSslRuntimeCapabilities(
            simulation_supported=True,
            live_agent_supported=False,
            live_server_supported=False,
        ),
    )

    federated_ssl_registry.register_federated_ssl_method_descriptor(
        "dummy_federated_ssl_method"
    )(dummy_descriptor)

    assert (
        federated_ssl_registry.resolve_federated_ssl_method_descriptor(
            "dummy_federated_ssl_method"
        )
        is dummy_descriptor
    )
    assert federated_ssl_registry.list_federated_ssl_method_descriptors() == (
        dummy_descriptor,
    )


def test_federated_ssl_required_views_must_be_non_empty_and_unique() -> None:
    with pytest.raises(ValueError, match="view_names must not be empty"):
        FederatedSslRequiredViews(
            view_names=(),
            view_generator_name="training_example_backend",
        )

    with pytest.raises(ValueError, match="view_names must be unique"):
        FederatedSslRequiredViews(
            view_names=("single_view", "single_view"),
            view_generator_name="training_example_backend",
        )


def test_local_update_profile_validates_training_objective_drift() -> None:
    profile = LocalUpdateProfile.from_mapping(
        {
            "algorithm_profile_name": "prototype_pseudo_label_v1",
            "training_scope": "adapter_only",
            "training_backend_name": "diagonal_scale_heuristic",
            "confidence_threshold": 0.6,
            "margin_threshold": 0.02,
            "example_generation_backend_name": "prototype_rescore",
            "evidence_backend_name": "prototype_similarity_evidence",
            "scorer_backend_name": "prototype_similarity",
            "score_policy_name": "max_cosine",
            "score_top_k": None,
            "pseudo_label_algorithm_name": "top1_margin_threshold",
            "acceptance_policy_name": "top1_margin_threshold",
            "privacy_guard_name": "diagonal_scale_clip_only",
            "evidence_backend_temperature": 1.0,
        }
    )
    objective = TrainingObjectiveConfig.from_mapping(
        {
            **profile.to_training_objective_mapping(),
            "adapter_family_specific.extra": "kept",
        }
    )

    require_training_objective_matches_local_update_profile(
        objective_config=objective,
        local_update_profile=profile,
    )

    drifted_objective = TrainingObjectiveConfig.from_mapping(
        {
            **profile.to_training_objective_mapping(),
            "privacy_guard_name": "noop",
        }
    )
    with pytest.raises(ValueError, match="local_update_profile"):
        require_training_objective_matches_local_update_profile(
            objective_config=drifted_objective,
            local_update_profile=profile,
        )


def test_fl_profile_compatibility_rejects_adapter_family_drift() -> None:
    profile = LocalUpdateProfile.from_mapping(
        {
            "algorithm_profile_name": "prototype_pseudo_label_v1",
            "training_scope": "adapter_only",
            "training_backend_name": "diagonal_scale_heuristic",
            "confidence_threshold": 0.6,
            "margin_threshold": 0.02,
            "example_generation_backend_name": "prototype_rescore",
            "evidence_backend_name": "prototype_similarity_evidence",
            "scorer_backend_name": "prototype_similarity",
            "score_policy_name": "max_cosine",
            "score_top_k": None,
            "pseudo_label_algorithm_name": "top1_margin_threshold",
            "acceptance_policy_name": "top1_margin_threshold",
            "privacy_guard_name": "diagonal_scale_clip_only",
            "evidence_backend_temperature": 1.0,
        }
    )

    with pytest.raises(ValueError, match="local_update_profile.*round_runtime_profile"):
        validate_federated_ssl_profile_compatibility(
            FederatedSslProfileCompatibilityContext(
                method_descriptor=FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
                local_update_profile=profile,
                local_update_adapter_kind="diagonal_scale",
                round_adapter_family_name="lora_classifier",
                round_aggregation_backend_name="fedavg",
            )
        )
