from __future__ import annotations

from methods.adaptation.diagonal_scale.config import (
    DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG,
)
from methods.federated_ssl.training_default_values import (
    DEFAULT_TRAINING_TASK_RUNTIME_DEFAULTS,
)
from methods.federated_ssl.training_defaults import (
    DEFAULT_TRAINING_PROFILE,
    PSEUDO_LABEL_SELF_TRAINING_V1_PROFILE,
    build_default_secure_aggregation_config,
    build_default_training_objective_config,
    build_default_training_selection_policy,
)


def test_default_training_profile_points_to_versioned_bundle() -> None:
    assert DEFAULT_TRAINING_PROFILE is PSEUDO_LABEL_SELF_TRAINING_V1_PROFILE
    assert DEFAULT_TRAINING_PROFILE.profile_name == "pseudo_label_self_training.v1"


def test_default_training_objective_builder_uses_shared_profile() -> None:
    config = build_default_training_objective_config()

    assert (
        config.training_backend_name == DEFAULT_TRAINING_PROFILE.training_backend_name
    )
    assert (
        config.algorithm_profile_name == DEFAULT_TRAINING_PROFILE.algorithm_profile_name
    )
    assert config.confidence_threshold == DEFAULT_TRAINING_PROFILE.confidence_threshold
    assert config.margin_threshold == DEFAULT_TRAINING_PROFILE.margin_threshold
    assert (
        config.example_generation_backend_name
        == DEFAULT_TRAINING_PROFILE.example_generation_backend_name
    )
    assert (
        config.evidence_backend_name
        == DEFAULT_TRAINING_PROFILE.evidence_backend_name
    )
    assert config.scorer_backend_name == DEFAULT_TRAINING_PROFILE.scorer_backend_name
    assert config.privacy_guard_name == DEFAULT_TRAINING_PROFILE.privacy_guard_name
    assert (
        config.pseudo_label_algorithm_name
        == DEFAULT_TRAINING_PROFILE.pseudo_label_algorithm_name
    )
    assert (
        config.acceptance_policy_name
        == DEFAULT_TRAINING_PROFILE.acceptance_policy_name
    )
    assert config.extras == {
        "training_backend.delta_scale_multiplier": (
            DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.delta_scale_multiplier
        ),
        "training_backend.max_abs_delta": (
            DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.max_abs_delta
        ),
        "training_backend.minimum_effective_scale": (
            DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.minimum_effective_scale
        ),
    }


def test_default_training_objective_builder_accepts_overrides() -> None:
    config = build_default_training_objective_config(
        overrides={
            "confidence_threshold": 0.75,
            "score_top_k": 3,
            "training_backend.max_abs_delta": 0.02,
        }
    )

    assert config.confidence_threshold == 0.75
    assert config.score_top_k == 3
    assert config.extras["training_backend.max_abs_delta"] == 0.02


def test_default_training_selection_policy_builder_uses_shared_profile() -> None:
    policy = build_default_training_selection_policy()

    assert policy.max_examples == DEFAULT_TRAINING_PROFILE.max_examples


def test_default_secure_aggregation_builder_starts_disabled() -> None:
    config = build_default_secure_aggregation_config()

    assert config.required is False


def test_default_training_profile_exposes_round_task_runtime_defaults() -> None:
    assert DEFAULT_TRAINING_PROFILE.local_epochs == 1
    assert DEFAULT_TRAINING_PROFILE.batch_size == 16
    assert DEFAULT_TRAINING_PROFILE.learning_rate == 1e-4
    assert DEFAULT_TRAINING_PROFILE.max_steps == 50
    assert (
        DEFAULT_TRAINING_PROFILE.task_runtime_defaults
        is DEFAULT_TRAINING_TASK_RUNTIME_DEFAULTS
    )
