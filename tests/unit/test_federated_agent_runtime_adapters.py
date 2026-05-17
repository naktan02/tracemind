"""FL simulation agent runtime adapter 단위 검증."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AGGREGATION_ARTIFACT_REF_PREFIX,
    AggregationArtifactStore,
)
from methods.adaptation.lora_classifier.config import (
    LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL,
    LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
    LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED,
    LoraClassifierTrainingBackendConfig,
)
from methods.federated_ssl.runtime_fallbacks import (
    RUNTIME_FALLBACK_TRAINING_PROFILE,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedLocalTrainerRuntimeConfig,
    FederatedQuerySslObjectiveConfig,
)
from scripts.runtime_adapters.federated_agent import (
    query_ssl_lora_classifier_trainer as qtrainer,
)
from scripts.runtime_adapters.federated_agent.backend_resolver import (
    resolve_example_generation_backend_name,
)
from scripts.runtime_adapters.federated_agent.query_ssl_lora_classifier_trainer import (
    _prepare_delta_materialization,
    run_query_ssl_lora_classifier_local_training,
    upload_agent_local_lora_classifier_update,
)
from scripts.runtime_adapters.federated_agent.row_validator import (
    require_rows_supported_by_example_backend,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
    make_lora_classifier_state_payload,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)


def test_resolve_example_generation_backend_name_uses_runtime_fallback() -> None:
    assert resolve_example_generation_backend_name(objective_config=None) == (
        RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
    )
    assert (
        resolve_example_generation_backend_name(
            objective_config=SimpleNamespace(example_generation_backend_name=None),
        )
        == RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
    )


def test_resolve_example_generation_backend_name_prefers_objective_config() -> None:
    assert (
        resolve_example_generation_backend_name(
            objective_config=SimpleNamespace(
                example_generation_backend_name="weak_strong_pair",
            ),
        )
        == "weak_strong_pair"
    )


def test_row_validator_rejects_missing_weak_strong_views() -> None:
    with pytest.raises(
        ValueError,
        match="requires each row to include both weak_text/strong_text",
    ):
        require_rows_supported_by_example_backend(
            rows=[{"query_id": "q1", "text": "panic", "weak_text": "panic weak"}],
            backend_name="weak_strong_pair",
        )


def test_row_validator_accepts_usb_text_and_augmentation_fields() -> None:
    require_rows_supported_by_example_backend(
        rows=[
            {
                "query_id": "q1",
                "text": "panic weak",
                "aug_0": "panic strong de",
                "aug_1": "panic strong fr",
            }
        ],
        backend_name="weak_strong_pair",
    )


def test_row_validator_rejects_partial_usb_augmentation_fields() -> None:
    with pytest.raises(
        ValueError,
        match="requires each row to include both weak_text/strong_text",
    ):
        require_rows_supported_by_example_backend(
            rows=[{"query_id": "q1", "text": "panic weak", "aug_0": "panic strong"}],
            backend_name="weak_strong_pair",
        )


def test_row_validator_accepts_non_multiview_backend_without_view_fields() -> None:
    require_rows_supported_by_example_backend(
        rows=[{"query_id": "q1", "text": "panic"}],
        backend_name="prototype_rescore",
    )


@pytest.mark.parametrize(
    ("method_name", "algorithm_name", "parameters"),
    [
        (
            "fixmatch_usb_v1",
            "fixmatch",
            {
                "temperature": 0.5,
                "p_cutoff": 0.95,
                "hard_label": True,
                "lambda_u": 1.0,
                "supervised_loss_weight": 1.0,
                "unlabeled_batch_size": 2,
            },
        ),
        (
            "flexmatch_usb_v1",
            "flexmatch",
            {
                "temperature": 0.5,
                "p_cutoff": 0.95,
                "hard_label": True,
                "thresh_warmup": True,
                "lambda_u": 1.0,
                "supervised_loss_weight": 1.0,
                "unlabeled_batch_size": 2,
            },
        ),
        (
            "freematch_usb_v1",
            "freematch",
            {
                "temperature": 0.5,
                "hard_label": True,
                "ema_p": 0.999,
                "ent_loss_ratio": 0.01,
                "use_quantile": False,
                "clip_thresh": False,
                "lambda_u": 1.0,
                "supervised_loss_weight": 1.0,
                "unlabeled_batch_size": 2,
            },
        ),
        (
            "pseudolabel_usb_v1",
            "pseudolabel",
            {
                "p_cutoff": 0.95,
                "unsup_warm_up": 0.4,
                "lambda_u": 1.0,
                "supervised_loss_weight": 1.0,
                "unlabeled_batch_size": 2,
            },
        ),
    ],
)
def test_query_ssl_lora_local_training_resolves_selected_ssl_algorithm(
    tmp_path,
    monkeypatch,
    method_name,
    algorithm_name,
    parameters,
) -> None:
    captured: dict[str, object] = {}
    lora_config = LoraClassifierTrainingBackendConfig(
        delta_format=LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
    )
    active_state = make_lora_classifier_state_payload(
        model_id="mxbai-lora-classifier",
        model_revision="sim_rev_0000",
        training_scope="adapter_only",
        backbone=lora_config.to_backbone_payload(),
        lora_config=lora_config.to_lora_config_payload(),
        label_schema=("anxiety", "normal"),
    )
    update_payload = make_lora_classifier_delta_payload(
        model_id="mxbai-lora-classifier",
        base_model_revision="sim_rev_0000",
        training_scope="adapter_only",
        backbone=lora_config.to_backbone_payload(),
        lora_config=lora_config.to_lora_config_payload(),
        label_schema=("anxiety", "normal"),
        example_count=1,
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1]},
        classifier_head_weight_deltas={"anxiety": [0.1], "normal": [-0.1]},
        classifier_head_bias_deltas={"anxiety": 0.01, "normal": -0.01},
        delta_format=LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
    )

    monkeypatch.setattr(
        qtrainer,
        "_build_lora_classifier_model",
        lambda **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(qtrainer, "_load_base_parameters", lambda **_kwargs: object())
    monkeypatch.setattr(
        qtrainer,
        "load_lora_classifier_base_parameters_into_model",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(qtrainer, "build_dataloader", lambda **_kwargs: [object()])
    monkeypatch.setattr(
        qtrainer,
        "_build_unlabeled_loader",
        lambda **_kwargs: [object()],
    )

    def _fake_train_query_ssl_classifier(**kwargs):
        captured["algorithm"] = kwargs["algorithm"]
        return kwargs["model"], [{"train_loss": 0.1}], {}

    monkeypatch.setattr(
        qtrainer,
        "train_query_ssl_classifier",
        _fake_train_query_ssl_classifier,
    )
    monkeypatch.setattr(
        qtrainer,
        "extract_lora_classifier_parameter_deltas",
        lambda **_kwargs: ({}, {}, {}),
    )
    monkeypatch.setattr(
        qtrainer,
        "build_query_ssl_lora_update_payload",
        lambda **_kwargs: SimpleNamespace(
            update_payload=update_payload,
            client_metrics={"query_ssl_local_steps": 1.0},
            accepted_unlabeled_count=1,
        ),
    )

    result = run_query_ssl_lora_classifier_local_training(
        client_id="agent_01",
        seed=42,
        output_dir=tmp_path,
        labeled_rows=[
            {
                "query_id": "l1",
                "text": "panic",
                "mapped_label_4": "anxiety",
            }
        ],
        unlabeled_rows=[
            {
                "query_id": "u1",
                "text": "weak text",
                "aug_0": "strong text de",
                "aug_1": "strong text fr",
                "mapped_label_4": "normal",
            }
        ],
        active_adapter_state=active_state,
        training_task=SimpleNamespace(
            round_id="round_0001",
            task_id="task_round_0001",
            training_scope="adapter_only",
            local_epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            max_steps=3,
            gradient_clip_norm=None,
            objective_config=TrainingObjectiveConfig.from_mapping(
                {"training_backend_name": "lora_classifier_trainer"}
            ),
            selection_policy=TrainingSelectionPolicy.from_mapping({"max_examples": 1}),
            task_type=TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
        ),
        model_manifest=SimpleNamespace(
            model_id="mxbai-lora-classifier",
            model_revision="sim_rev_0000",
        ),
        query_ssl_config=FederatedQuerySslObjectiveConfig(
            method_name=method_name,
            algorithm_name=algorithm_name,
            parameters=parameters,
            strong_view_policy="first_aug",
            unlabeled_batch_size=2,
        ),
        lora_config=lora_config,
        trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(
            device="cpu",
            local_files_only=True,
        ),
    )

    assert captured["algorithm"].algorithm_name == algorithm_name
    if algorithm_name == "flexmatch":
        assert captured["algorithm"].thresh_warmup is True
    assert result.update_payload == update_payload


def test_query_ssl_lora_delta_materialization_writes_server_owned_refs(
    tmp_path,
) -> None:
    plan = _prepare_delta_materialization(
        output_dir=tmp_path,
        update_id="update_round_0001_agent_01_test",
        training_task=SimpleNamespace(round_id="round_0001"),
        client_id="agent_01",
        delta_format=LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED,
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={
            "anxiety": [0.3, -0.1],
            "normal": [-0.3, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
    )

    assert plan.delta_format == LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED
    assert plan.include_inline_deltas is False
    assert plan.lora_delta_artifact_ref is not None
    assert plan.classifier_head_delta_artifact_ref is not None
    assert plan.lora_delta_artifact_ref.startswith(AGGREGATION_ARTIFACT_REF_PREFIX)
    assert plan.classifier_head_delta_artifact_ref.startswith(
        AGGREGATION_ARTIFACT_REF_PREFIX
    )

    store = AggregationArtifactStore(
        state_root=tmp_path / "main_server" / "aggregation_artifacts"
    )
    lora_artifact = store.load_json_artifact(artifact_ref=plan.lora_delta_artifact_ref)
    head_artifact = store.load_json_artifact(
        artifact_ref=plan.classifier_head_delta_artifact_ref
    )
    assert lora_artifact["schema_version"] == (
        "lora_classifier_client_delta_artifact.v1"
    )
    assert lora_artifact["lora_parameter_deltas"] == {
        "encoder.q_proj.lora_A": [0.1, -0.2]
    }
    assert head_artifact["schema_version"] == (
        "lora_classifier_client_head_delta_artifact.v1"
    )
    assert head_artifact["classifier_head_bias_deltas"] == {
        "anxiety": 0.05,
        "normal": -0.05,
    }


def test_query_ssl_lora_delta_materialization_keeps_inline_debug_payload(
    tmp_path,
) -> None:
    plan = _prepare_delta_materialization(
        output_dir=tmp_path,
        update_id="update_round_0001_agent_01_test",
        training_task=SimpleNamespace(round_id="round_0001"),
        client_id="agent_01",
        delta_format=LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={"anxiety": [0.3, -0.1]},
        classifier_head_bias_deltas={"anxiety": 0.05},
    )

    assert plan.delta_format == LORA_CLASSIFIER_DELTA_FORMAT_INLINE
    assert plan.include_inline_deltas is True
    assert plan.lora_delta_artifact_ref is None
    assert plan.classifier_head_delta_artifact_ref is None
    assert not (tmp_path / "main_server" / "aggregation_artifacts").exists()


def test_query_ssl_lora_delta_materialization_writes_agent_local_refs(
    tmp_path,
) -> None:
    plan = _prepare_delta_materialization(
        output_dir=tmp_path,
        update_id="update_round_0001_agent_01_test",
        training_task=SimpleNamespace(round_id="round_0001"),
        client_id="agent_01",
        delta_format=LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL,
        artifact_ref_prefix="agent-local://lora_classifier",
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={
            "anxiety": [0.3, -0.1],
            "normal": [-0.3, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
    )

    assert plan.delta_format == LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL
    assert plan.include_inline_deltas is False
    assert plan.lora_delta_artifact_ref is not None
    assert plan.classifier_head_delta_artifact_ref is not None
    assert plan.lora_delta_artifact_ref.startswith("agent-local://")
    assert plan.classifier_head_delta_artifact_ref.startswith("agent-local://")
    local_artifacts = sorted(
        (tmp_path / "agents" / "local_artifacts" / "versions").glob("**/*.json")
    )
    assert len(local_artifacts) == 2
    assert not (tmp_path / "main_server" / "aggregation_artifacts").exists()


def test_upload_agent_local_lora_update_materializes_server_owned_refs(
    tmp_path,
) -> None:
    plan = _prepare_delta_materialization(
        output_dir=tmp_path,
        update_id="update_round_0001_agent_01_test",
        training_task=SimpleNamespace(round_id="round_0001"),
        client_id="agent_01",
        delta_format=LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL,
        artifact_ref_prefix="agent-local://lora_classifier",
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={
            "anxiety": [0.3, -0.1],
            "normal": [-0.3, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
    )
    update_payload = make_lora_classifier_delta_payload(
        model_id="mxbai-lora-classifier",
        base_model_revision="sim_rev_0000",
        training_scope="adapter_only",
        backbone={
            "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "backbone_revision": "main",
            "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "tokenizer_revision": "main",
            "pooling": "mean",
            "max_length": 256,
            "task_prefix": "",
        },
        lora_config={
            "peft_adapter_name": "lora",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "bias": "none",
            "target_modules": "all-linear",
            "use_rslora": False,
        },
        label_schema=["anxiety", "normal"],
        example_count=2,
        lora_delta_artifact_ref=plan.lora_delta_artifact_ref,
        classifier_head_delta_artifact_ref=plan.classifier_head_delta_artifact_ref,
        delta_format=LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL,
    )

    uploaded = upload_agent_local_lora_classifier_update(
        output_dir=tmp_path,
        update_payload=update_payload,
    )

    assert uploaded.delta_format == LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED
    assert uploaded.lora_delta_artifact_ref is not None
    assert uploaded.classifier_head_delta_artifact_ref is not None
    assert uploaded.lora_delta_artifact_ref.startswith(AGGREGATION_ARTIFACT_REF_PREFIX)
    assert uploaded.classifier_head_delta_artifact_ref.startswith(
        AGGREGATION_ARTIFACT_REF_PREFIX
    )
    store = AggregationArtifactStore(
        state_root=tmp_path / "main_server" / "aggregation_artifacts"
    )
    lora_artifact = store.load_json_artifact(
        artifact_ref=uploaded.lora_delta_artifact_ref
    )
    head_artifact = store.load_json_artifact(
        artifact_ref=uploaded.classifier_head_delta_artifact_ref
    )
    assert lora_artifact["lora_parameter_deltas"] == {
        "encoder.q_proj.lora_A": [0.1, -0.2]
    }
    assert head_artifact["classifier_head_bias_deltas"] == {
        "anxiety": 0.05,
        "normal": -0.05,
    }
