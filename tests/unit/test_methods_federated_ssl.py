"""Reusable FL SSL method descriptor tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from methods.common.registry import MethodRegistry
from methods.federated_ssl import registry as federated_ssl_registry
from methods.federated_ssl.base import (
    FederatedSslLocalStepSpec,
    FederatedSslMethodDescriptor,
    FederatedSslMethodRecipe,
    FederatedSslRequiredViews,
    FederatedSslRoundStateExchangeSpec,
    FederatedSslRuntimeCapabilities,
    FederatedSslRuntimePair,
    FederatedSslServerStepSpec,
)
from methods.federated_ssl.compatibility import (
    FederatedSslProfileCompatibilityContext,
    validate_federated_ssl_profile_compatibility,
)
from methods.federated_ssl.execution_plan import (
    COMPOSITION_MODE_MANUAL,
    COMPOSITION_MODE_METHOD_OWNED,
    SECURITY_POLICY_PLAINTEXT,
    build_federated_ssl_execution_plan,
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

TEST_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
METHOD_OWNED_RECIPE = FederatedSslMethodRecipe(
    method_name="method_owned_ssl",
    supported_local_update_profile_names=("prototype_pseudo_label_v1",),
    supported_runtime_pairs=(
        FederatedSslRuntimePair(
            adapter_family_name="diagonal_scale",
            aggregation_backend_name="fedavg",
        ),
    ),
)
METHOD_OWNED_DESCRIPTOR = FederatedSslMethodDescriptor(
    name="method_owned_ssl",
    implementation_status="test_only",
    required_views=FederatedSslRequiredViews(
        view_names=("single_view",),
        view_generator_name="training_example_backend",
    ),
    local_step=FederatedSslLocalStepSpec(
        step_name="pseudo_label_self_training",
        client_trainer_name="local_training_service",
        pseudo_labeler_name="ssl_pseudo_label_selection_hook",
        training_row_source="unlabeled_pool_when_available",
    ),
    server_step=FederatedSslServerStepSpec(
        server_aggregator_name="round_runtime_aggregation_backend",
        round_policy_name="round_active_pair_only",
        server_aggregate_hint="use_round_runtime_aggregation_backend",
    ),
    round_state_exchange=FederatedSslRoundStateExchangeSpec(exchange_name="none"),
    runtime_capabilities=FederatedSslRuntimeCapabilities(
        simulation_supported=True,
        live_agent_supported=False,
        live_server_supported=False,
    ),
    recipe=METHOD_OWNED_RECIPE,
)


def _load_test_only_federated_ssl_method_fixture() -> object:
    module_name = "_tracemind_test_only_federated_ssl_method"
    module_path = TEST_FIXTURE_ROOT / "federated_ssl_dummy_method.py"
    sys.modules.pop(module_name, None)
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Failed to load test fixture module: {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


def test_federated_ssl_descriptor_registry_has_no_manual_baseline_builtin() -> None:
    with pytest.raises(NotImplementedError, match="descriptor is not wired yet"):
        resolve_federated_ssl_method_descriptor("fedavg_pseudo_label")

    assert list_federated_ssl_method_descriptors() == ()


def test_federated_ssl_execution_plan_defaults_to_method_owned_plaintext() -> None:
    plan = build_federated_ssl_execution_plan(
        fl_method=None,
        security_policy=None,
        method_descriptor=METHOD_OWNED_DESCRIPTOR,
    )

    assert plan.method_name == "method_owned_ssl"
    assert plan.descriptor_name == "method_owned_ssl"
    assert plan.composition_mode == COMPOSITION_MODE_METHOD_OWNED
    assert plan.execution_role == "method_owned"
    assert plan.manual_axes.is_configured is False
    assert plan.round_state_exchange_name == "none"
    assert plan.required_client_metric_keys == ()
    assert plan.security_policy.name == SECURITY_POLICY_PLAINTEXT


def test_federated_ssl_execution_plan_supports_manual_lower_axes() -> None:
    plan = build_federated_ssl_execution_plan(
        fl_method={
            "composition_mode": COMPOSITION_MODE_MANUAL,
            "manual_axes": {
                "client_ssl_objective": "pseudo_label",
                "server_aggregation": "fedavg",
                "update_family": "diagonal_scale",
            },
        },
        security_policy={"name": "plaintext"},
        method_descriptor=None,
    )

    assert plan.method_name == "manual"
    assert plan.descriptor_name is None
    assert plan.composition_mode == COMPOSITION_MODE_MANUAL
    assert plan.execution_role == "manual_baseline"
    assert plan.manual_axes.client_ssl_objective == "pseudo_label"
    assert plan.manual_axes.server_aggregation == "fedavg"
    assert plan.manual_axes.update_family == "diagonal_scale"


def test_federated_ssl_execution_plan_rejects_method_owned_manual_axes() -> None:
    with pytest.raises(ValueError, match="manual_axes"):
        build_federated_ssl_execution_plan(
            fl_method={
                "name": "method_owned_ssl",
                "composition_mode": COMPOSITION_MODE_METHOD_OWNED,
                "manual_axes": {
                    "client_ssl_objective": "fixmatch",
                },
            },
            security_policy=None,
            method_descriptor=METHOD_OWNED_DESCRIPTOR,
        )


def test_federated_ssl_execution_plan_rejects_unsupported_security_policy() -> None:
    with pytest.raises(ValueError, match="Unsupported security_policy.name"):
        build_federated_ssl_execution_plan(
            fl_method=None,
            security_policy={"name": "secure_aggregation_v1"},
            method_descriptor=METHOD_OWNED_DESCRIPTOR,
        )


def test_federated_ssl_descriptor_registry_rejects_unwired_method() -> None:
    with pytest.raises(NotImplementedError, match="descriptor is not wired yet"):
        resolve_federated_ssl_method_descriptor("paper_method_candidate")


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

    fixture = _load_test_only_federated_ssl_method_fixture()
    dummy_descriptor = fixture.DUMMY_FEDERATED_SSL_DESCRIPTOR

    local_update_profile = LocalUpdateProfile.from_mapping(
        {
            "algorithm_profile_name": "dummy_local_update_profile_v1",
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

    assert (
        federated_ssl_registry.resolve_federated_ssl_method_descriptor(
            "dummy_metric_weighted_ssl"
        )
        is dummy_descriptor
    )
    assert federated_ssl_registry.list_federated_ssl_method_descriptors() == (
        dummy_descriptor,
    )
    assert dummy_descriptor.round_state_exchange is not None
    assert dummy_descriptor.round_state_exchange.exchange_name == (
        "client_metric_summary"
    )
    assert dummy_descriptor.requires_custom_server_runtime is True

    validate_federated_ssl_profile_compatibility(
        FederatedSslProfileCompatibilityContext(
            method_descriptor=dummy_descriptor,
            local_update_profile=local_update_profile,
            local_update_adapter_kind="diagonal_scale",
            round_adapter_family_name="diagonal_scale",
            round_aggregation_backend_name="fedavg",
        )
    )


def test_builtin_federated_ssl_registry_excludes_test_only_extension() -> None:
    descriptors = list_federated_ssl_method_descriptors()

    assert descriptors == ()


def test_federated_ssl_method_package_does_not_keep_manual_baseline_descriptor() -> (
    None
):
    method_root = Path(__file__).resolve().parents[2] / "methods" / "federated_ssl"

    assert not (method_root / "fedavg_pseudo_label" / "descriptor.py").exists()


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


def test_federated_ssl_local_step_rejects_unknown_training_row_source() -> None:
    with pytest.raises(ValueError, match="training_row_source"):
        FederatedSslLocalStepSpec(
            step_name="pseudo_label_self_training",
            client_trainer_name="local_training_service",
            pseudo_labeler_name="ssl_pseudo_label_selection_hook",
            training_row_source="unknown_rows",
        )


def test_federated_ssl_method_descriptor_rejects_recipe_name_drift() -> None:
    with pytest.raises(ValueError, match="same method name"):
        FederatedSslMethodDescriptor(
            name="method_owned_ssl",
            implementation_status="test_only",
            required_views=FederatedSslRequiredViews(
                view_names=("single_view",),
                view_generator_name="training_example_backend",
            ),
            local_step=FederatedSslLocalStepSpec(
                step_name="pseudo_label_self_training",
                client_trainer_name="local_training_service",
                pseudo_labeler_name="ssl_pseudo_label_selection_hook",
            ),
            server_step=FederatedSslServerStepSpec(
                server_aggregator_name="round_runtime_aggregation_backend",
                round_policy_name="round_active_pair_only",
                server_aggregate_hint="use_round_runtime_aggregation_backend",
            ),
            runtime_capabilities=FederatedSslRuntimeCapabilities(
                simulation_supported=True,
                live_agent_supported=False,
                live_server_supported=False,
            ),
            recipe=FederatedSslMethodRecipe(
                method_name="other_method",
                supported_runtime_pairs=(
                    FederatedSslRuntimePair(
                        adapter_family_name="diagonal_scale",
                        aggregation_backend_name="fedavg",
                    ),
                ),
            ),
        )


def test_federated_ssl_round_state_exchange_spec_rejects_duplicate_metric_keys() -> (
    None
):
    with pytest.raises(ValueError, match="required_client_metric_keys must be unique"):
        FederatedSslRoundStateExchangeSpec(
            exchange_name="client_metric_summary",
            required_client_metric_keys=("mean_confidence", "MEAN_CONFIDENCE"),
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


def test_fl_ssl_compatibility_rejects_adapter_family_drift() -> None:
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

    with pytest.raises(ValueError, match="local_update_profile.*round_runtime"):
        validate_federated_ssl_profile_compatibility(
            FederatedSslProfileCompatibilityContext(
                method_descriptor=METHOD_OWNED_DESCRIPTOR,
                local_update_profile=profile,
                local_update_adapter_kind="diagonal_scale",
                round_adapter_family_name="lora_classifier",
                round_aggregation_backend_name="fedavg",
            )
        )


def test_fl_ssl_compatibility_rejects_method_recipe_mismatch() -> None:
    profile = LocalUpdateProfile.from_mapping(
        {
            "algorithm_profile_name": "paper_candidate_profile",
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

    with pytest.raises(ValueError, match="method recipe.*local_update_profile"):
        validate_federated_ssl_profile_compatibility(
            FederatedSslProfileCompatibilityContext(
                method_descriptor=METHOD_OWNED_DESCRIPTOR,
                local_update_profile=profile,
                local_update_adapter_kind="diagonal_scale",
                round_adapter_family_name="diagonal_scale",
                round_aggregation_backend_name="fedavg",
            )
        )

    with pytest.raises(ValueError, match="method recipe.*round runtime pair"):
        validate_federated_ssl_profile_compatibility(
            FederatedSslProfileCompatibilityContext(
                method_descriptor=METHOD_OWNED_DESCRIPTOR,
                local_update_profile=None,
                local_update_adapter_kind="diagonal_scale",
                round_adapter_family_name="diagonal_scale",
                round_aggregation_backend_name="trimmed_mean",
            )
        )
