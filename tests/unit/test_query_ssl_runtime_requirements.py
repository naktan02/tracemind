"""Query SSL runtime capability descriptor 검증."""

from __future__ import annotations

from methods.ssl.base import (
    QUERY_SSL_ALGORITHM_STATE_ADAPTIVE_THRESHOLD,
    QUERY_SSL_ALGORITHM_STATE_DATASET_STATE,
    QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA,
    QUERY_SSL_ALGORITHM_STATE_FEATURE_QUEUE,
    QUERY_SSL_ALGORITHM_STATE_PROBABILITY_QUEUE,
    QUERY_SSL_ALGORITHM_STATE_STATELESS,
    QUERY_SSL_ALGORITHM_STATE_STEP_COUNTER,
    QUERY_SSL_ALGORITHM_STATE_WEIGHTING_EMA,
    QUERY_SSL_BATCH_SURFACE_WEAK_ONLY,
    QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
    QUERY_SSL_BATCH_SURFACE_WEAK_STRONG_PAIR,
    QUERY_SSL_INPUT_TRANSFORM_NONE,
    QUERY_SSL_MODEL_OUTPUT_LOGITS,
    QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_AUXILIARY_TRAINABLE_MODULE,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
    QUERY_SSL_TEACHER_STATE_NONE,
    QuerySslRuntimeRequirements,
)
from methods.ssl.registry import resolve_query_ssl_algorithm_descriptor


def test_runtime_requirements_accept_future_capability_strings() -> None:
    requirements = QuerySslRuntimeRequirements(
        batch_surface="future_batch_surface",
        model_outputs=frozenset({"future_model_output"}),
        algorithm_state_surface=frozenset({"future_state_surface"}),
        input_transform_surface="future_transform",
        optimizer_lifecycle=frozenset({"future_optimizer_lifecycle"}),
        teacher_state="future_teacher_state",
        step_context_required=True,
    )

    assert requirements.batch_surface == "future_batch_surface"
    assert requirements.model_outputs == frozenset({"future_model_output"})
    assert requirements.algorithm_state_surface == frozenset({"future_state_surface"})
    assert requirements.input_transform_surface == "future_transform"
    assert requirements.optimizer_lifecycle == frozenset({"future_optimizer_lifecycle"})
    assert requirements.teacher_state == "future_teacher_state"
    assert requirements.step_context_required is True


def test_builtin_descriptors_expose_current_runtime_capabilities() -> None:
    expected_state_surfaces = {
        "fixmatch": frozenset({QUERY_SSL_ALGORITHM_STATE_STATELESS}),
        "flexmatch": frozenset(
            {
                QUERY_SSL_ALGORITHM_STATE_ADAPTIVE_THRESHOLD,
                QUERY_SSL_ALGORITHM_STATE_DATASET_STATE,
            }
        ),
        "adamatch": frozenset({QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA}),
        "freematch": frozenset(
            {
                QUERY_SSL_ALGORITHM_STATE_ADAPTIVE_THRESHOLD,
                QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA,
            }
        ),
        "pseudo_label": frozenset({QUERY_SSL_ALGORITHM_STATE_STEP_COUNTER}),
        "comatch": frozenset(
            {
                QUERY_SSL_ALGORITHM_STATE_FEATURE_QUEUE,
                QUERY_SSL_ALGORITHM_STATE_PROBABILITY_QUEUE,
            }
        ),
        "softmatch": frozenset(
            {
                QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA,
                QUERY_SSL_ALGORITHM_STATE_WEIGHTING_EMA,
            }
        ),
    }
    expected_batch_surfaces = {
        "fixmatch": QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
        "flexmatch": QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
        "adamatch": QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
        "freematch": QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
        "pseudo_label": QUERY_SSL_BATCH_SURFACE_WEAK_ONLY,
        "comatch": QUERY_SSL_BATCH_SURFACE_WEAK_STRONG_PAIR,
        "softmatch": QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
    }

    for algorithm_name, expected_state_surface in expected_state_surfaces.items():
        descriptor = resolve_query_ssl_algorithm_descriptor(algorithm_name)
        requirements = descriptor.runtime_requirements

        assert requirements.batch_surface == expected_batch_surfaces[algorithm_name]
        expected_model_outputs = frozenset({QUERY_SSL_MODEL_OUTPUT_LOGITS})
        if algorithm_name == "comatch":
            expected_model_outputs = frozenset(
                {
                    QUERY_SSL_MODEL_OUTPUT_LOGITS,
                    QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
                }
            )
        assert requirements.model_outputs == expected_model_outputs
        assert requirements.algorithm_state_surface == expected_state_surface
        assert requirements.input_transform_surface == QUERY_SSL_INPUT_TRANSFORM_NONE
        expected_optimizer_lifecycle = frozenset(
            {QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP}
        )
        if algorithm_name == "comatch":
            expected_optimizer_lifecycle = frozenset(
                {
                    QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
                    QUERY_SSL_OPTIMIZER_LIFECYCLE_AUXILIARY_TRAINABLE_MODULE,
                }
            )
        assert requirements.optimizer_lifecycle == expected_optimizer_lifecycle
        assert requirements.teacher_state == QUERY_SSL_TEACHER_STATE_NONE
