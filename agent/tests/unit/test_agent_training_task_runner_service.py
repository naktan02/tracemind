"""Agent training task runner service tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from agent.src.infrastructure.repositories.captured_text_repository import (
    CAPTURED_TEXT_VIEW_STATUS_READY,
    CapturedTextGeneratedViewRecord,
    CapturedTextRecord,
    CapturedTextRepository,
)
from agent.src.services.federation.rounds.runtime_service import (
    FederationRunResult,
    FederationRunStatus,
)
from agent.src.services.training.execution.agent_training_task_runner_service import (
    AgentTrainingTaskRunnerService,
    AgentTrainingTaskRunRequest,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_peft_classifier_state_payload,
)
from shared.src.contracts.model_contracts import make_embedding_manifest
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
)


def _build_task_payload() -> TrainingTaskPayload:
    return TrainingTaskPayload(
        schema_version="training_task.v1",
        task_id="task_multiview",
        round_id="round_multiview",
        model_id="tracemind-embed",
        model_revision="rev_multiview",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=4,
        objective_config=TrainingObjectiveConfigPayload(
            algorithm_profile_name="prototype_pseudo_label_v1",
            training_backend_name="peft_classifier_trainer",
            confidence_threshold=0.6,
            margin_threshold=0.02,
            example_generation_backend_name="weak_strong_pair",
            evidence_backend_name="prototype_similarity_evidence",
            scorer_backend_name="prototype_similarity",
            acceptance_policy_name="top1_margin_threshold",
            privacy_guard_name="noop",
        ),
        selection_policy=TrainingSelectionPolicyPayload(),
    )


def _build_query_ssl_task_payload() -> TrainingTaskPayload:
    return _build_task_payload().model_copy(
        update={
            "task_id": "task_query_ssl",
            "objective_config": TrainingObjectiveConfigPayload(
                algorithm_profile_name="peft_pseudo_label_v1",
                training_backend_name="peft_classifier_trainer",
                confidence_threshold=0.6,
                margin_threshold=0.02,
                example_generation_backend_name="weak_strong_pair",
                evidence_backend_name="prototype_similarity_evidence",
                scorer_backend_name="prototype_similarity",
                acceptance_policy_name="top1_margin_threshold",
                privacy_guard_name="noop",
                extras={
                    "query_ssl.method_name": "fixmatch_usb_v1",
                    "query_ssl.algorithm_name": "fixmatch",
                    "query_ssl.strong_view_policy": "first_aug",
                    "query_ssl.unlabeled_batch_size": 8,
                    "query_ssl.temperature": 0.5,
                    "query_ssl.p_cutoff": 0.95,
                    "query_ssl.hard_label": True,
                    "query_ssl.lambda_u": 1.0,
                    "query_ssl.supervised_loss_weight": 1.0,
                },
            ),
        }
    )


def _build_supported_task_payload() -> TrainingTaskPayload:
    return TrainingTaskPayload(
        schema_version="training_task.v1",
        task_id="task_001",
        round_id="round_0001",
        model_id="tracemind-embed",
        model_revision="rev_001",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=4,
        objective_config=TrainingObjectiveConfigPayload(
            training_backend_name="peft_classifier_trainer",
            example_generation_backend_name="prototype_rescore",
            evidence_backend_name="prototype_similarity_evidence",
            scorer_backend_name="prototype_similarity",
            acceptance_policy_name="top1_margin_threshold",
            privacy_guard_name="noop",
        ),
        selection_policy=TrainingSelectionPolicyPayload(),
    )


def _build_prototype_pack() -> PrototypePackPayload:
    return PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_001",
            "embedding_model_id": "tracemind-embed",
            "embedding_model_revision": "rev_001",
            "mapping_version": "ourafla_to_4cat.v1",
            "build_method": "mean_centroid_l2_normalized",
            "distance_metric": "cosine",
            "built_at": datetime(2026, 4, 2, tzinfo=timezone.utc),
            "categories": {
                "anxiety": [
                    {
                        "prototype_id": "anxiety:single",
                        "centroid": [1.0, 0.0],
                        "sample_count": 2,
                    }
                ]
            },
        }
    )


def _peft_backbone() -> dict[str, object]:
    return {
        "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "backbone_revision": "main",
        "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "tokenizer_revision": "main",
        "pooling": "mean",
        "max_length": 256,
        "task_prefix": "",
    }


def _peft_adapter_config() -> dict[str, object]:
    return {
        "peft_adapter_name": "lora",
        "parameters": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "bias": "none",
            "target_modules": "all-linear",
            "use_rslora": False,
        },
    }


def _build_peft_state(*, model_revision: str):
    return make_peft_classifier_state_payload(
        model_id="tracemind-embed",
        model_revision=model_revision,
        backbone=_peft_backbone(),
        peft_adapter_config=_peft_adapter_config(),
        label_schema=["anxiety", "normal"],
    )


def _build_service(
    *,
    repo: MagicMock,
    proto_service: MagicMock,
    proto_sync_service: MagicMock,
    shared_adapter_runtime_service: MagicMock,
    shared_adapter_sync_service: MagicMock,
    round_client_factory: MagicMock,
    runtime_factory: MagicMock,
    captured_text_repository: CapturedTextRepository | None = None,
    embedding_adapter: object | None = None,
    query_ssl_task_service: object | None = None,
) -> AgentTrainingTaskRunnerService:
    kwargs = {}
    if query_ssl_task_service is not None:
        kwargs["query_ssl_task_service"] = query_ssl_task_service
    return AgentTrainingTaskRunnerService(
        scored_event_repository=repo,
        prototype_runtime_service=proto_service,
        prototype_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        federation_runtime_service_factory=runtime_factory,
        captured_text_repository=captured_text_repository,
        embedding_adapter=embedding_adapter,  # type: ignore[arg-type]
        **kwargs,
    )


class StubEmbeddingAdapter:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _text in texts]


def test_runner_builds_multiview_examples_from_captured_text_views(
    tmp_path: Path,
) -> None:
    repo = MagicMock()
    repo.get_recent_stored.return_value = ()
    captured_repo = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    occurred_at = datetime(2026, 6, 7, 1, 0, tzinfo=timezone.utc)
    captured_repo.save(
        CapturedTextRecord(
            event_id="event_generated",
            occurred_at=occurred_at,
            received_at=occurred_at,
            text="불안해",
            locale="ko",
            source_type="search",
            surface_type="search_box",
        )
    )
    stored = captured_repo.get("event_generated")
    assert stored is not None
    captured_repo.save_generated_view(
        CapturedTextGeneratedViewRecord(
            event_id="event_generated",
            generated_at=occurred_at,
            weak_text="I feel anxious",
            strong_text_0="I am feeling anxious",
            strong_text_1="I feel worried",
            generator_name="unit-test",
            generator_version="v1",
            source_text_fingerprint=stored.text_fingerprint,
            metadata={"weak_text_translated": True},
        )
    )
    captured_repo.mark_view_generation_status(
        event_id="event_generated",
        status=CAPTURED_TEXT_VIEW_STATUS_READY,
    )
    proto_service = MagicMock()
    proto_service.get_active_pack.return_value = _build_prototype_pack()
    proto_sync_service = MagicMock()
    shared_adapter_sync_service = MagicMock()
    active_manifest = make_embedding_manifest(
        model_id="tracemind-embed",
        model_revision="rev_multiview",
        auxiliary_artifact_versions={"prototype_pack": "proto_001"},
        artifact_ref="/server/state/rev_multiview.json",
    )
    active_state = MagicMock()
    active_state.apply.side_effect = lambda embedding: list(embedding)
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_runtime_service.get_active_manifest.return_value = active_manifest
    shared_adapter_runtime_service.get_active_state.return_value = active_state
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    federation_runtime = MagicMock()
    federation_runtime.run_current_task.return_value = FederationRunResult(
        status=FederationRunStatus.INSUFFICIENT_EXAMPLES,
        round_id="round_multiview",
        task_id="task_multiview",
        example_count=1,
    )
    runtime_factory = MagicMock(return_value=federation_runtime)
    service = _build_service(
        repo=repo,
        proto_service=proto_service,
        proto_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
        captured_text_repository=captured_repo,
        embedding_adapter=StubEmbeddingAdapter(),
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == str(FederationRunStatus.INSUFFICIENT_EXAMPLES)
    proto_sync_service.pull_version.assert_called_once_with(
        server_base_url="http://server.test",
        prototype_version="proto_001",
    )
    call_kwargs = federation_runtime.run_current_task.call_args.kwargs
    training_examples = call_kwargs["training_examples"]
    assert len(training_examples) == 1
    assert training_examples[0].metadata["weak_text"] == "I feel anxious"
    assert training_examples[0].metadata["strong_text"] == "I am feeling anxious"


def test_runner_reports_missing_embedding_adapter_for_captured_text_views(
    tmp_path: Path,
) -> None:
    repo = MagicMock()
    repo.get_recent_stored.return_value = ()
    captured_repo = CapturedTextRepository(db_path=tmp_path / "captured_text.db")
    occurred_at = datetime(2026, 6, 7, 1, 0, tzinfo=timezone.utc)
    captured_repo.save(
        CapturedTextRecord(
            event_id="event_generated",
            occurred_at=occurred_at,
            received_at=occurred_at,
            text="불안해",
            locale="ko",
            source_type="search",
            surface_type="search_box",
        )
    )
    stored = captured_repo.get("event_generated")
    assert stored is not None
    captured_repo.save_generated_view(
        CapturedTextGeneratedViewRecord(
            event_id="event_generated",
            generated_at=occurred_at,
            weak_text="I feel anxious",
            strong_text_0="I am feeling anxious",
            strong_text_1="I feel worried",
            generator_name="unit-test",
            generator_version="v1",
            source_text_fingerprint=stored.text_fingerprint,
            metadata={"weak_text_translated": True},
        )
    )
    captured_repo.mark_view_generation_status(
        event_id="event_generated",
        status=CAPTURED_TEXT_VIEW_STATUS_READY,
    )
    proto_service = MagicMock()
    proto_service.get_active_pack.return_value = _build_prototype_pack()
    proto_sync_service = MagicMock()
    shared_adapter_sync_service = MagicMock()
    active_manifest = make_embedding_manifest(
        model_id="tracemind-embed",
        model_revision="rev_multiview",
        auxiliary_artifact_versions={"prototype_pack": "proto_001"},
        artifact_ref="/server/state/rev_multiview.json",
    )
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_runtime_service.get_active_manifest.return_value = active_manifest
    shared_adapter_runtime_service.get_active_state.return_value = MagicMock()
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    runtime_factory = MagicMock()
    service = _build_service(
        repo=repo,
        proto_service=proto_service,
        proto_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
        captured_text_repository=captured_repo,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == "missing_embedding_adapter"
    runtime_factory.assert_not_called()


def test_runner_routes_query_ssl_task_to_query_ssl_service() -> None:
    repo = MagicMock()
    proto_service = MagicMock()
    proto_sync_service = MagicMock()
    shared_adapter_sync_service = MagicMock()
    active_manifest = make_embedding_manifest(
        model_id="tracemind-embed",
        model_revision="rev_multiview",
        artifact_ref="/server/state/rev_multiview.json",
    )
    active_state = _build_peft_state(model_revision="rev_multiview")
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_runtime_service.get_active_manifest.return_value = active_manifest
    shared_adapter_runtime_service.get_active_state.return_value = active_state
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_query_ssl_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    runtime_factory = MagicMock()
    query_ssl_task_service = MagicMock()
    query_ssl_task_service.run_current_task.return_value = FederationRunResult(
        status=FederationRunStatus.UPLOADED,
        round_id="round_multiview",
        task_id="task_query_ssl",
        update_id="update_query_ssl",
        example_count=3,
        accepted_count=2,
        message="Query SSL update 업로드 완료.",
    )
    service = _build_service(
        repo=repo,
        proto_service=proto_service,
        proto_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
        query_ssl_task_service=query_ssl_task_service,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == str(FederationRunStatus.UPLOADED)
    assert response.round_id == "round_multiview"
    assert response.task_id == "task_query_ssl"
    assert response.update_id == "update_query_ssl"
    shared_adapter_sync_service.pull_current.assert_called_once_with(
        server_base_url="http://server.test"
    )
    query_ssl_request = query_ssl_task_service.run_current_task.call_args.args[0]
    assert query_ssl_request.training_task.task_id == "task_query_ssl"
    assert query_ssl_request.model_manifest is active_manifest
    assert query_ssl_request.active_state is active_state
    runtime_factory.assert_not_called()


def test_runner_uses_empty_examples_when_multiview_has_no_generated_views() -> None:
    repo = MagicMock()
    repo.get_recent_stored.return_value = ()
    proto_service = MagicMock()
    proto_service.get_active_pack.return_value = _build_prototype_pack()
    proto_sync_service = MagicMock()
    shared_adapter_sync_service = MagicMock()
    active_manifest = make_embedding_manifest(
        model_id="tracemind-embed",
        model_revision="rev_multiview",
        auxiliary_artifact_versions={"prototype_pack": "proto_001"},
        artifact_ref="/server/state/rev_multiview.json",
    )
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_runtime_service.get_active_manifest.return_value = active_manifest
    shared_adapter_runtime_service.get_active_state.return_value = MagicMock()
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    federation_runtime = MagicMock()
    federation_runtime.run_current_task.return_value = FederationRunResult(
        status=FederationRunStatus.INSUFFICIENT_EXAMPLES,
        round_id="round_multiview",
        task_id="task_multiview",
    )
    runtime_factory = MagicMock(return_value=federation_runtime)
    service = _build_service(
        repo=repo,
        proto_service=proto_service,
        proto_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == str(FederationRunStatus.INSUFFICIENT_EXAMPLES)
    call_kwargs = federation_runtime.run_current_task.call_args.kwargs
    assert call_kwargs["training_examples"] == ()


def test_runner_syncs_shared_state_and_uses_matching_manifest() -> None:
    repo = MagicMock()
    repo.get_recent_stored.return_value = ()
    proto_service = MagicMock()
    proto_service.get_active_pack.return_value = _build_prototype_pack()
    proto_sync_service = MagicMock()
    shared_adapter_sync_service = MagicMock()
    active_manifest = make_embedding_manifest(
        model_id="tracemind-embed",
        model_revision="rev_001",
        auxiliary_artifact_versions={"prototype_pack": "proto_001"},
        artifact_ref="/server/state/rev_001.json",
    )
    active_state = _build_peft_state(model_revision="rev_001")
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_runtime_service.get_active_manifest.return_value = active_manifest
    shared_adapter_runtime_service.get_active_state.return_value = active_state
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_supported_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    federation_runtime = MagicMock()
    federation_runtime.run_current_task.return_value = FederationRunResult(
        status=FederationRunStatus.INSUFFICIENT_EXAMPLES,
        round_id="round_0001",
        task_id="task_001",
    )
    runtime_factory = MagicMock(return_value=federation_runtime)
    service = _build_service(
        repo=repo,
        proto_service=proto_service,
        proto_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == str(FederationRunStatus.INSUFFICIENT_EXAMPLES)
    shared_adapter_sync_service.pull_current.assert_called_once_with(
        server_base_url="http://server.test"
    )
    proto_sync_service.pull_version.assert_called_once_with(
        server_base_url="http://server.test",
        prototype_version="proto_001",
    )
    call_kwargs = federation_runtime.run_current_task.call_args.kwargs
    assert call_kwargs["model_manifest"].model_revision == "rev_001"
    assert call_kwargs["task_payload"].model_revision == "rev_001"


def test_runner_does_not_pull_prototype_when_manifest_has_no_auxiliary_pack() -> None:
    repo = MagicMock()
    repo.get_recent_stored.return_value = ()
    proto_service = MagicMock()
    proto_sync_service = MagicMock()
    shared_adapter_sync_service = MagicMock()
    active_manifest = make_embedding_manifest(
        model_id="tracemind-embed",
        model_revision="rev_001",
        artifact_ref="/server/state/rev_001.json",
    )
    active_state = _build_peft_state(model_revision="rev_001")
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_runtime_service.get_active_manifest.return_value = active_manifest
    shared_adapter_runtime_service.get_active_state.return_value = active_state
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_supported_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    federation_runtime = MagicMock()
    federation_runtime.run_current_task.return_value = FederationRunResult(
        status=FederationRunStatus.INSUFFICIENT_EXAMPLES,
        round_id="round_0001",
        task_id="task_001",
    )
    runtime_factory = MagicMock(return_value=federation_runtime)
    service = _build_service(
        repo=repo,
        proto_service=proto_service,
        proto_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == str(FederationRunStatus.INSUFFICIENT_EXAMPLES)
    proto_sync_service.pull_version.assert_not_called()
    proto_service.get_active_pack.assert_not_called()
    call_kwargs = federation_runtime.run_current_task.call_args.kwargs
    assert call_kwargs["training_examples"] == ()
