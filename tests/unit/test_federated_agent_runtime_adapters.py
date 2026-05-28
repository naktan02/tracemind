"""FL simulation agent runtime adapter 단위 검증."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AGGREGATION_ARTIFACT_REF_PREFIX,
    AggregationArtifactStore,
)
from methods.adaptation.peft_text_classifier.config import (
    PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
    PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL,
    PEFT_ENCODER_DELTA_FORMAT_INLINE,
    PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED,
    LoraClassifierTrainingBackendConfig,
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_classifier.federated_ssl import (
    helper_provider,
)
from methods.adaptation.peft_text_classifier.training import (
    query_ssl_local_training as qcore,
)
from methods.adaptation.peft_text_classifier.training_backend import (
    PeftEncoderTrainingBackend,
)
from methods.adaptation.peft_text_classifier.update import (
    merged_tensor_artifact as merged_artifacts,
)
from methods.adaptation.peft_text_classifier.update import (
    partitioned_tensor_artifact as partitioned_artifacts,
)
from methods.adaptation.peft_text_classifier.update.delta_artifacts import (
    PeftEncoderDeltaMaterializer,
    upload_agent_local_peft_encoder_update,
)
from methods.adaptation.peft_text_classifier.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.adaptation.peft_text_classifier.update.partitioned_delta import (
    PeftEncoderPartitionDelta,
)
from methods.common.timing import TimingRecorder
from methods.evaluation.pseudo_label_quality import PseudoLabelQualitySummary
from methods.federated_ssl.capability_axes import (
    LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
    LOCAL_SSL_POLICY_FIXMATCH,
)
from methods.federated_ssl.runtime_fallbacks import (
    RUNTIME_FALLBACK_TRAINING_PROFILE,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedLocalTrainerRuntimeConfig,
    FederatedQuerySslObjectiveConfig,
)
from scripts.experiments.fl_ssl.federated_simulation.runtime_resources import (
    RoundBaseSnapshotCache,
)
from scripts.runtime_adapters.federated_agent import base_state_materialization
from scripts.runtime_adapters.federated_agent import (
    peft_encoder_local_training as qtrainer,
)
from scripts.runtime_adapters.federated_agent.artifact_store import (
    SimulationClientArtifactStore,
)
from scripts.runtime_adapters.federated_agent.backend_resolver import (
    resolve_example_generation_backend_name,
)
from scripts.runtime_adapters.federated_agent.base_state_materialization import (
    load_peft_encoder_base_parameters,
)
from scripts.runtime_adapters.federated_agent.peft_encoder_local_training import (
    run_query_ssl_peft_encoder_local_training,
)
from scripts.runtime_adapters.federated_agent.row_validator import (
    require_rows_supported_by_example_backend,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
    make_lora_classifier_state_payload,
    make_peft_classifier_delta_payload,
    make_peft_classifier_state_payload,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)

build_peft_encoder_helper_provider_for_local_ssl_policy = (
    helper_provider.build_peft_encoder_helper_provider_for_local_ssl_policy
)


def test_peft_training_backend_owns_simulation_inline_executor_wiring() -> None:
    backend = PeftEncoderTrainingBackend(
        config=PeftEncoderTrainingBackendConfig(
            delta_format=PEFT_ENCODER_DELTA_FORMAT_INLINE
        )
    )

    runtime_backend = backend.with_simulation_inline_train_executor()

    assert runtime_backend.train_executor is not None
    assert runtime_backend.backend_name == backend.backend_name
    assert runtime_backend.payload_format == backend.payload_format
    assert runtime_backend.adapter_kind == backend.adapter_kind


def test_peft_training_backend_leaves_non_inline_simulation_backend_unchanged() -> None:
    backend = PeftEncoderTrainingBackend(
        config=PeftEncoderTrainingBackendConfig(
            delta_format=PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL
        )
    )

    assert backend.with_simulation_inline_train_executor() is backend


def prepare_delta_materialization(*, output_dir, **kwargs):
    return PeftEncoderDeltaMaterializer(
        artifact_store=SimulationClientArtifactStore(output_dir=output_dir)
    ).prepare(**kwargs)


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


def test_peft_encoder_base_parameters_cache_lora_v1_state(
    tmp_path,
    monkeypatch,
) -> None:
    active_state = make_lora_classifier_state_payload(
        model_id="mxbai-peft-classifier",
        model_revision="sim_rev_0003",
        training_scope="adapter_only",
        backbone=LoraClassifierTrainingBackendConfig().to_backbone_payload(),
        lora_config=LoraClassifierTrainingBackendConfig().to_lora_config_payload(),
        label_schema=("anxiety", "normal"),
        lora_adapter_artifact_ref="server-aggregate://sim_rev_0003/lora_adapter",
        classifier_head_artifact_ref="server-aggregate://sim_rev_0003/head",
    )
    materialized = PeftEncoderMaterializedState(
        lora_parameters={"lora.test": [0.1]},
        classifier_head_weights={
            "anxiety": [0.2, 0.0],
            "normal": [0.0, -0.2],
        },
        classifier_head_biases={"anxiety": 0.01, "normal": -0.01},
    )
    calls = {"count": 0}

    def _fake_materialize(**_kwargs):
        calls["count"] += 1
        return materialized

    monkeypatch.setattr(
        base_state_materialization,
        "_materialize_peft_encoder_base_parameters",
        _fake_materialize,
    )
    cache = RoundBaseSnapshotCache()

    first = load_peft_encoder_base_parameters(
        active_adapter_state=active_state,
        output_dir=tmp_path,
        aggregated_at=SimpleNamespace(),
        round_base_snapshot_cache=cache,
    )
    second = load_peft_encoder_base_parameters(
        active_adapter_state=active_state,
        output_dir=tmp_path,
        aggregated_at=SimpleNamespace(),
        round_base_snapshot_cache=cache,
    )

    assert first is materialized
    assert second is materialized
    assert calls["count"] == 1
    assert cache.miss_count == 1
    assert cache.hit_count == 1


def test_local_ssl_helper_provider_resolver_skips_non_helper_policy(
    monkeypatch,
) -> None:
    calls = {"count": 0}

    def _fake_builder(**_kwargs):
        calls["count"] += 1
        return object()

    monkeypatch.setattr(
        "methods.adaptation.peft_text_classifier.federated_ssl.helper_provider."
        "build_peft_encoder_helper_probability_provider",
        _fake_builder,
    )

    provider = build_peft_encoder_helper_provider_for_local_ssl_policy(
        method_name="fedmatch",
        local_ssl_policy_name=LOCAL_SSL_POLICY_FIXMATCH,
        peer_context=None,
        peer_snapshots=None,
        labels=("anxiety", "normal"),
        lora_config=PeftEncoderTrainingBackendConfig(),
        trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(device="cpu"),
        runtime_resource_cache=None,
        timing_recorder=None,
    )

    assert provider is None
    assert calls["count"] == 0


def test_local_ssl_helper_provider_resolver_builds_fedmatch_provider(
    monkeypatch,
) -> None:
    provider = object()
    captured: dict[str, object] = {}

    def _fake_builder(**kwargs):
        captured.update(kwargs)
        return provider

    monkeypatch.setattr(
        "methods.adaptation.peft_text_classifier.federated_ssl.helper_provider."
        "build_peft_encoder_helper_probability_provider",
        _fake_builder,
    )
    timing = TimingRecorder()
    runtime_cache = object()

    resolved = build_peft_encoder_helper_provider_for_local_ssl_policy(
        method_name="fedmatch",
        local_ssl_policy_name=LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
        peer_context=None,
        peer_snapshots={},
        labels=("anxiety", "normal"),
        lora_config=PeftEncoderTrainingBackendConfig(),
        trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(device="cpu"),
        runtime_resource_cache=runtime_cache,
        timing_recorder=timing,
    )

    assert resolved is provider
    assert captured["labels"] == ("anxiety", "normal")
    assert captured["runtime_resource_cache"] is runtime_cache
    assert "adapter_helper_provider_prepare_seconds" in timing.to_mapping()


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
def test_query_ssl_peft_encoder_local_training_resolves_selected_ssl_algorithm(
    tmp_path,
    monkeypatch,
    method_name,
    algorithm_name,
    parameters,
) -> None:
    captured: dict[str, object] = {}
    peft_config = PeftEncoderTrainingBackendConfig(
        delta_format=PEFT_ENCODER_DELTA_FORMAT_INLINE,
    )
    active_state = make_peft_classifier_state_payload(
        model_id="mxbai-peft-classifier",
        model_revision="sim_rev_0000",
        training_scope="adapter_only",
        backbone=peft_config.to_backbone_payload(),
        peft_adapter_config=peft_config.to_peft_adapter_config_payload(),
        label_schema=("anxiety", "normal"),
    )
    update_payload = make_peft_classifier_delta_payload(
        model_id="mxbai-peft-classifier",
        base_model_revision="sim_rev_0000",
        training_scope="adapter_only",
        backbone=peft_config.to_backbone_payload(),
        peft_adapter_config=peft_config.to_peft_adapter_config_payload(),
        label_schema=("anxiety", "normal"),
        example_count=1,
        peft_parameter_deltas={"encoder.q_proj.lora_A": [0.1]},
        classifier_head_weight_deltas={"anxiety": [0.1], "normal": [-0.1]},
        classifier_head_bias_deltas={"anxiety": 0.01, "normal": -0.01},
        delta_format=PEFT_ENCODER_DELTA_FORMAT_INLINE,
    )
    runtime_resource_cache = object()

    def _fake_build_peft_encoder_model(**kwargs):
        captured["runtime_resource_cache"] = kwargs["runtime_resource_cache"]
        return object(), object()

    monkeypatch.setattr(
        qcore,
        "_build_peft_encoder_model",
        _fake_build_peft_encoder_model,
    )
    monkeypatch.setattr(
        qtrainer,
        "load_peft_encoder_base_parameters",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        qcore,
        "load_peft_encoder_base_parameters_into_model",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(qcore, "build_dataloader", lambda **_kwargs: [object()])
    monkeypatch.setattr(
        qcore,
        "_build_unlabeled_loader",
        lambda **_kwargs: [object()],
    )

    def _fake_train_query_ssl_classifier(**kwargs):
        captured["algorithm"] = kwargs["algorithm"]
        return kwargs["model"], [{"train_loss": 0.1}], {}

    monkeypatch.setattr(
        qcore,
        "train_query_ssl_classifier",
        _fake_train_query_ssl_classifier,
    )
    monkeypatch.setattr(
        qcore,
        "extract_peft_encoder_parameter_deltas",
        lambda **_kwargs: ({}, {}, {}),
    )
    monkeypatch.setattr(
        qcore,
        "build_query_ssl_peft_encoder_update_payload",
        lambda **_kwargs: SimpleNamespace(
            update_payload=update_payload,
            client_metrics={"query_ssl_local_steps": 1.0},
            accepted_unlabeled_count=1,
        ),
    )
    monkeypatch.setattr(
        qcore,
        "build_final_snapshot_pseudo_label_quality",
        lambda **_kwargs: PseudoLabelQualitySummary(
            pseudo_label_confidence_mean=0.97,
            pseudo_label_margin_mean=0.5,
            pseudo_label_correct_count=1,
            pseudo_label_evaluated_count=1,
            accepted_label_distribution={"normal": 1},
            rejected_label_distribution={},
        ),
    )

    result = run_query_ssl_peft_encoder_local_training(
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
        training_task=TrainingTask(
            schema_version="training_task.v1",
            round_id="round_0001",
            task_id="task_round_0001",
            model_id="mxbai-peft-classifier",
            model_revision="sim_rev_0000",
            training_scope="adapter_only",
            local_epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            max_steps=3,
            gradient_clip_norm=None,
            objective_config=TrainingObjectiveConfig.from_mapping(
                {"training_backend_name": PEFT_CLASSIFIER_TRAINING_BACKEND_NAME}
            ),
            selection_policy=TrainingSelectionPolicy.from_mapping({"max_examples": 1}),
            task_type=TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
        ),
        model_manifest=SimpleNamespace(
            model_id="mxbai-peft-classifier",
            model_revision="sim_rev_0000",
        ),
        query_ssl_config=FederatedQuerySslObjectiveConfig(
            method_name=method_name,
            algorithm_name=algorithm_name,
            parameters=parameters,
            strong_view_policy="first_aug",
            unlabeled_batch_size=2,
        ),
        lora_config=peft_config,
        trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(
            device="cpu",
            local_files_only=True,
        ),
        runtime_resource_cache=runtime_resource_cache,
    )

    assert captured["algorithm"].algorithm_name == algorithm_name
    assert captured["runtime_resource_cache"] is runtime_resource_cache
    if algorithm_name == "flexmatch":
        assert captured["algorithm"].thresh_warmup is True
    assert result.update_payload == update_payload
    assert result.pseudo_label_quality.pseudo_label_correct_count == 1


def test_query_ssl_peft_encoder_delta_materialization_writes_server_owned_refs(
    tmp_path,
) -> None:
    plan = prepare_delta_materialization(
        output_dir=tmp_path,
        update_id="update_round_0001_agent_01_test",
        training_task=SimpleNamespace(round_id="round_0001"),
        client_id="agent_01",
        delta_format=PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED,
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={
            "anxiety": [0.3, -0.1],
            "normal": [-0.3, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
    )

    assert plan.delta_format == PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED
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
    lora_tensors, lora_metadata = store.load_safetensors_artifact(
        artifact_ref=plan.lora_delta_artifact_ref
    )
    head_tensors, head_metadata = store.load_safetensors_artifact(
        artifact_ref=plan.classifier_head_delta_artifact_ref
    )
    lora_index_key = merged_artifacts.LORA_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY
    head_index_key = merged_artifacts.HEAD_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY
    assert lora_index_key in lora_metadata
    assert head_index_key in head_metadata
    for artifact_ref in (
        plan.lora_delta_artifact_ref,
        plan.classifier_head_delta_artifact_ref,
    ):
        artifact_id = store.artifact_id_from_ref(artifact_ref)
        assert artifact_id is not None
        assert store.path_for_safetensors_artifact(artifact_id).exists()
        assert not store.path_for_artifact(artifact_id).exists()

    lora_deltas = merged_artifacts.parse_lora_delta_tensor_artifact(
        tensors=lora_tensors,
        metadata=lora_metadata,
    )
    head_weight_deltas, head_bias_deltas = (
        merged_artifacts.parse_classifier_head_delta_tensor_artifact(
            tensors=head_tensors,
            metadata=head_metadata,
        )
    )
    assert lora_deltas["encoder.q_proj.lora_A"] == pytest.approx([0.1, -0.2])
    assert head_weight_deltas["anxiety"] == pytest.approx([0.3, -0.1])
    assert head_bias_deltas == pytest.approx(
        {
            "anxiety": 0.05,
            "normal": -0.05,
        }
    )


def test_query_ssl_peft_encoder_delta_materialization_writes_partitioned_ref(
    tmp_path,
) -> None:
    plan = prepare_delta_materialization(
        output_dir=tmp_path,
        update_id="update_round_0001_agent_01_test",
        training_task=SimpleNamespace(round_id="round_0001"),
        client_id="agent_01",
        delta_format=PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED,
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={"anxiety": [0.3, -0.1]},
        classifier_head_bias_deltas={"anxiety": 0.05},
        partitioned_deltas={
            "sigma": PeftEncoderPartitionDelta(
                partition_name="sigma",
                lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1]},
                classifier_head_weight_deltas={"anxiety": [0.2]},
                classifier_head_bias_deltas={"anxiety": 0.03},
            )
        },
        materialize_primary_deltas=False,
    )

    assert plan.lora_delta_artifact_ref is None
    assert plan.classifier_head_delta_artifact_ref is None
    assert plan.partitioned_deltas_artifact_ref is not None
    assert plan.partitioned_deltas_artifact_ref.startswith(
        AGGREGATION_ARTIFACT_REF_PREFIX
    )
    store = AggregationArtifactStore(
        state_root=tmp_path / "main_server" / "aggregation_artifacts"
    )
    tensors, metadata = store.load_safetensors_artifact(
        artifact_ref=plan.partitioned_deltas_artifact_ref
    )
    partitioned_index_key = (
        partitioned_artifacts.PARTITIONED_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY
    )
    assert partitioned_index_key in metadata
    artifact_id = store.artifact_id_from_ref(plan.partitioned_deltas_artifact_ref)
    assert artifact_id is not None
    assert store.path_for_safetensors_artifact(artifact_id).exists()
    assert not store.path_for_artifact(artifact_id).exists()
    partitions = partitioned_artifacts.parse_partitioned_delta_tensor_artifact(
        tensors=tensors,
        metadata=metadata,
    )
    assert set(partitions) == {"sigma"}
    assert partitions["sigma"].lora_parameter_deltas[
        "encoder.q_proj.lora_A"
    ] == pytest.approx([0.1])
    assert partitions["sigma"].classifier_head_weight_deltas["anxiety"] == (
        pytest.approx([0.2])
    )
    assert partitions["sigma"].classifier_head_bias_deltas == pytest.approx(
        {"anxiety": 0.03}
    )


def test_query_ssl_peft_encoder_delta_materialization_keeps_inline_debug_payload(
    tmp_path,
) -> None:
    plan = prepare_delta_materialization(
        output_dir=tmp_path,
        update_id="update_round_0001_agent_01_test",
        training_task=SimpleNamespace(round_id="round_0001"),
        client_id="agent_01",
        delta_format=PEFT_ENCODER_DELTA_FORMAT_INLINE,
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={"anxiety": [0.3, -0.1]},
        classifier_head_bias_deltas={"anxiety": 0.05},
    )

    assert plan.delta_format == PEFT_ENCODER_DELTA_FORMAT_INLINE
    assert plan.include_inline_deltas is True
    assert plan.lora_delta_artifact_ref is None
    assert plan.classifier_head_delta_artifact_ref is None
    assert not (tmp_path / "main_server" / "aggregation_artifacts").exists()


def test_query_ssl_peft_encoder_delta_materialization_requires_prefix_for_agent_local(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="requires artifact_ref_prefix"):
        prepare_delta_materialization(
            output_dir=tmp_path,
            update_id="update_round_0001_agent_01_test",
            training_task=SimpleNamespace(round_id="round_0001"),
            client_id="agent_01",
            delta_format=PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL,
            lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
            classifier_head_weight_deltas={"anxiety": [0.3, -0.1]},
            classifier_head_bias_deltas={"anxiety": 0.05},
        )


def test_query_ssl_peft_encoder_delta_materialization_writes_agent_local_refs(
    tmp_path,
) -> None:
    plan = prepare_delta_materialization(
        output_dir=tmp_path,
        update_id="update_round_0001_agent_01_test",
        training_task=SimpleNamespace(round_id="round_0001"),
        client_id="agent_01",
        delta_format=PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL,
        artifact_ref_prefix="agent-local://peft_classifier",
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={
            "anxiety": [0.3, -0.1],
            "normal": [-0.3, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
    )

    assert plan.delta_format == PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL
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


def test_upload_agent_local_lora_v1_update_materializes_server_owned_refs(
    tmp_path,
) -> None:
    plan = prepare_delta_materialization(
        output_dir=tmp_path,
        update_id="update_round_0001_agent_01_test",
        training_task=SimpleNamespace(round_id="round_0001"),
        client_id="agent_01",
        delta_format=PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL,
        artifact_ref_prefix="agent-local://lora_classifier",
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={
            "anxiety": [0.3, -0.1],
            "normal": [-0.3, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
    )
    update_payload = make_lora_classifier_delta_payload(
        model_id="mxbai-peft-classifier",
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
        delta_format=PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL,
    )

    uploaded = upload_agent_local_peft_encoder_update(
        artifact_store=SimulationClientArtifactStore(output_dir=tmp_path),
        update_payload=update_payload,
    )

    assert uploaded.delta_format == PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED
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


def test_upload_agent_local_peft_update_materializes_server_owned_refs(
    tmp_path,
) -> None:
    plan = prepare_delta_materialization(
        output_dir=tmp_path,
        update_id="update_round_0001_agent_01_test",
        training_task=SimpleNamespace(round_id="round_0001"),
        client_id="agent_01",
        delta_format=PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL,
        artifact_ref_prefix="agent-local://peft_classifier",
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={
            "anxiety": [0.3, -0.1],
            "normal": [-0.3, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
    )
    update_payload = make_peft_classifier_delta_payload(
        model_id="mxbai-peft-classifier",
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
        peft_adapter_config={
            "peft_adapter_name": "lora",
            "parameters": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.1,
                "bias": "none",
                "target_modules": "all-linear",
                "use_rslora": False,
            },
        },
        label_schema=["anxiety", "normal"],
        example_count=2,
        peft_adapter_delta_artifact_ref=plan.lora_delta_artifact_ref,
        classifier_head_delta_artifact_ref=plan.classifier_head_delta_artifact_ref,
        delta_format=PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL,
    )

    uploaded = upload_agent_local_peft_encoder_update(
        artifact_store=SimulationClientArtifactStore(output_dir=tmp_path),
        update_payload=update_payload,
    )

    assert uploaded.delta_format == PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED
    assert uploaded.peft_adapter_delta_artifact_ref is not None
    assert uploaded.classifier_head_delta_artifact_ref is not None
    assert uploaded.peft_adapter_delta_artifact_ref.startswith(
        AGGREGATION_ARTIFACT_REF_PREFIX
    )
    assert uploaded.classifier_head_delta_artifact_ref.startswith(
        AGGREGATION_ARTIFACT_REF_PREFIX
    )
