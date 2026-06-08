from __future__ import annotations

from methods.federated_ssl.runtime_fallbacks import (
    FIXMATCH_FEDAVG_V1_RUNTIME_FALLBACK,
    RUNTIME_FALLBACK_TRAINING_PROFILE,
    RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS,
    build_runtime_fallback_secure_aggregation_config,
    build_runtime_fallback_training_objective_config,
    build_runtime_fallback_training_selection_policy,
)


def test_runtime_fallback_profile_points_to_versioned_bundle() -> None:
    assert RUNTIME_FALLBACK_TRAINING_PROFILE is FIXMATCH_FEDAVG_V1_RUNTIME_FALLBACK


def test_runtime_fallback_objective_builder_uses_runtime_fallback_profile() -> None:
    config = build_runtime_fallback_training_objective_config()

    assert config.to_mapping() == dict(
        RUNTIME_FALLBACK_TRAINING_PROFILE.objective_mapping
    )


def test_runtime_fallback_objective_builder_accepts_overrides() -> None:
    config = build_runtime_fallback_training_objective_config(
        overrides={
            "query_ssl.p_cutoff": 0.75,
            "training_backend.max_abs_delta": 0.02,
        }
    )

    assert config.extras["query_ssl.p_cutoff"] == 0.75
    assert config.extras["training_backend.max_abs_delta"] == 0.02


def test_runtime_fallback_selection_policy_builder_uses_profile() -> None:
    policy = build_runtime_fallback_training_selection_policy()

    assert policy.max_examples == RUNTIME_FALLBACK_TRAINING_PROFILE.max_examples


def test_runtime_fallback_secure_aggregation_builder_starts_disabled() -> None:
    config = build_runtime_fallback_secure_aggregation_config()

    assert config.required is False


def test_runtime_fallback_profile_exposes_round_task_runtime_defaults() -> None:
    assert (
        RUNTIME_FALLBACK_TRAINING_PROFILE.local_epochs
        == RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS.local_epochs
    )
    assert (
        RUNTIME_FALLBACK_TRAINING_PROFILE.batch_size
        == RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS.batch_size
    )
    assert (
        RUNTIME_FALLBACK_TRAINING_PROFILE.learning_rate
        == RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS.learning_rate
    )
    assert (
        RUNTIME_FALLBACK_TRAINING_PROFILE.max_steps
        == RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS.max_steps
    )
    assert (
        RUNTIME_FALLBACK_TRAINING_PROFILE.task_runtime_defaults
        is RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS
    )
