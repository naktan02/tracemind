"""Agent Query SSL training task service tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from agent.src.infrastructure.repositories.captured_text_repository import (
    CAPTURED_TEXT_VIEW_STATUS_READY,
    CapturedTextGeneratedViewRecord,
    CapturedTextRecord,
    CapturedTextRepository,
)
from agent.src.infrastructure.repositories.training_usage_ledger_repository import (
    TRAINING_USAGE_ROLE_LABELED_ANCHOR,
    TRAINING_USAGE_ROLE_UNLABELED_GENERATED_VIEW,
    TrainingUsageLedgerRepository,
)
from agent.src.services.training_runtime.current_task.query_ssl_training_task_service import (  # noqa: E501
    AgentQuerySslTrainingTaskRunRequest,
    AgentQuerySslTrainingTaskService,
)
from methods.adaptation.peft_text_encoder import config as peft_config
from methods.adaptation.peft_text_encoder.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.query_text_views.local_training_budget import (
    build_query_ssl_local_step_plan,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_peft_classifier_delta_payload,
    make_peft_classifier_state_payload,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    PeftClassifierDelta,
)
from shared.src.contracts.model_contracts import make_embedding_manifest
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
    make_training_update_envelope,
)
from shared.src.domain.entities.inference.events import AnalysisEvent

QuerySslPeftEncoderClientTrainingResult = (
    qssl_training.QuerySslPeftEncoderClientTrainingResult
)


class _QuerySslPeftBackend:
    backend_name = "peft_classifier_trainer"
    config = peft_config.PeftEncoderTrainingBackendConfig()

    def __init__(self, update_payload: PeftClassifierDelta) -> None:
        self.update_payload = update_payload
        self.captured_kwargs: dict[str, object] | None = None

    def matches_objective_config(self, objective_config: object | None) -> bool:
        del objective_config
        return True

    def build_query_ssl_update(
        self,
        request: object,
    ) -> QuerySslPeftEncoderClientTrainingResult:
        local_session = request.local_session
        self.captured_kwargs = {
            "labels": tuple(local_session.labels),
            "labeled_rows": tuple(local_session.labeled_rows),
            "unlabeled_rows": tuple(local_session.unlabeled_rows),
        }
        training_task = local_session.training_task
        model_manifest = request.model_manifest
        update_envelope = make_training_update_envelope(
            update_id="update_query_ssl_test",
            round_id=training_task.round_id,
            task_id=training_task.task_id,
            model_id=model_manifest.model_id,
            base_model_revision=model_manifest.model_revision,
            training_scope=training_task.training_scope,
            payload_ref="client-submission::update_query_ssl_test",
            payload_format=PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
            example_count=self.update_payload.example_count,
            client_metrics={"query_ssl_local_steps": 1.0},
        )
        return QuerySslPeftEncoderClientTrainingResult(
            update_envelope=update_envelope,
            update_payload=self.update_payload,
            candidate_count=2,
            accepted_count=1,
            local_step_plan=build_query_ssl_local_step_plan(
                labeled_loader_steps=1,
                unlabeled_loader_steps=1,
                uses_labeled_batches=True,
                local_epochs=1,
                max_steps=1,
            ),
            client_metrics=update_envelope.client_metrics,
        )


def test_query_ssl_training_task_service_uploads_query_ssl_update(tmp_path) -> None:
    occurred_at = datetime(2026, 6, 7, 1, 0, tzinfo=timezone.utc)
    scored_repo = AnalysisEventRepository(db_path=tmp_path / "scored.db")
    scored_repo.save(
        AnalysisEvent(
            query_id="labeled_1",
            occurred_at=occurred_at,
            translated_text="I feel anxious",
            embedding_model_id="embed",
            translation_model_id="nllb",
            category_scores={"anxiety": 0.9, "normal": 0.1},
        )
    )
    captured_repo = CapturedTextRepository(db_path=tmp_path / "captured.db")
    captured_repo.save(
        CapturedTextRecord(
            event_id="unlabeled_1",
            occurred_at=occurred_at,
            received_at=occurred_at,
            text="불안해",
            locale="ko",
            source_type="search",
            surface_type="search_box",
        )
    )
    stored = captured_repo.get("unlabeled_1")
    assert stored is not None
    captured_repo.save_generated_view(
        CapturedTextGeneratedViewRecord(
            event_id="unlabeled_1",
            generated_at=occurred_at,
            weak_text="I am anxious",
            strong_text_0="I feel anxious now",
            strong_text_1="I am worried now",
            generator_name="unit-test",
            generator_version="v1",
            source_text_fingerprint=stored.text_fingerprint,
            metadata={"weak_text_translated": True},
        )
    )
    captured_repo.mark_view_generation_status(
        event_id="unlabeled_1",
        status=CAPTURED_TEXT_VIEW_STATUS_READY,
    )
    active_state = make_peft_classifier_state_payload(
        model_id="tracemind-embed",
        model_revision="rev_001",
        training_scope="adapter_only",
        backbone=peft_config.PeftEncoderTrainingBackendConfig().to_backbone_payload(),
        peft_adapter_config=(
            peft_config.PeftEncoderTrainingBackendConfig().to_peft_adapter_config_payload()
        ),
        label_schema=("anxiety", "normal"),
        updated_at=occurred_at,
    )
    update_payload = make_peft_classifier_delta_payload(
        model_id="tracemind-embed",
        base_model_revision="rev_001",
        training_scope="adapter_only",
        backbone=peft_config.PeftEncoderTrainingBackendConfig().to_backbone_payload(),
        peft_adapter_config=(
            peft_config.PeftEncoderTrainingBackendConfig().to_peft_adapter_config_payload()
        ),
        label_schema=("anxiety", "normal"),
        example_count=1,
        peft_parameter_deltas={"lora.test": [0.1]},
        classifier_head_weight_deltas={"anxiety": [0.1], "normal": [-0.1]},
        classifier_head_bias_deltas={"anxiety": 0.01, "normal": -0.01},
        delta_format="inline_delta",
    )
    backend = _QuerySslPeftBackend(update_payload)
    round_client = MagicMock()
    usage_ledger = TrainingUsageLedgerRepository(db_path=tmp_path / "usage.db")
    service = AgentQuerySslTrainingTaskService(
        backend=backend,
        usage_ledger_repository=usage_ledger,
    )

    result = service.run_current_task(
        AgentQuerySslTrainingTaskRunRequest(
            training_task=_query_ssl_task(),
            model_manifest=make_embedding_manifest(
                model_id="tracemind-embed",
                model_revision="rev_001",
                training_scope="adapter_only",
                artifact_ref="/tmp/rev_001.json",
            ),
            active_state=active_state,
            round_client=round_client,
            analysis_event_repository=scored_repo,
            captured_text_repository=captured_repo,
            analysis_event_days=7,
            agent_id="agent_01",
        )
    )

    assert result.status == "uploaded"
    assert result.update_id == "update_query_ssl_test"
    assert backend.captured_kwargs is not None
    assert backend.captured_kwargs["labels"] == ("anxiety", "normal")
    assert backend.captured_kwargs["labeled_rows"][0]["mapped_label_4"] == "anxiety"
    assert backend.captured_kwargs["unlabeled_rows"][0]["aug_0"] == (
        "I feel anxious now"
    )
    round_client.upload_update.assert_called_once()
    usage_run = usage_ledger.get_run("update_query_ssl_test")
    usage_rows = usage_ledger.get_rows_for_update("update_query_ssl_test")
    assert usage_run is not None
    assert usage_run.round_id == "round_001"
    assert usage_run.task_id == "task_query_ssl"
    assert usage_run.agent_id == "agent_01"
    assert usage_run.objective_method_name == "fixmatch_usb_v1"
    assert usage_run.objective_algorithm_name == "fixmatch"
    assert usage_run.metadata["effective_method_family"] == "query_ssl"
    assert usage_run.metadata["fssl_method"] is None
    assert usage_run.metadata["query_ssl_method_name"] == "fixmatch_usb_v1"
    assert usage_run.candidate_count == 2
    assert usage_run.accepted_count == 1
    assert {(row.source_id, row.role) for row in usage_rows} == {
        ("labeled_1", TRAINING_USAGE_ROLE_LABELED_ANCHOR),
        ("unlabeled_1", TRAINING_USAGE_ROLE_UNLABELED_GENERATED_VIEW),
    }


def test_query_ssl_training_task_service_routes_fedmatch_method_owned_runtime(
    tmp_path,
) -> None:
    occurred_at = datetime(2026, 6, 7, 1, 0, tzinfo=timezone.utc)
    scored_repo = AnalysisEventRepository(db_path=tmp_path / "scored.db")
    scored_repo.save(
        AnalysisEvent(
            query_id="labeled_1",
            occurred_at=occurred_at,
            translated_text="I feel anxious",
            embedding_model_id="embed",
            translation_model_id="nllb",
            category_scores={"anxiety": 0.9, "normal": 0.1},
        )
    )
    captured_repo = CapturedTextRepository(db_path=tmp_path / "captured.db")
    captured_repo.save(
        CapturedTextRecord(
            event_id="unlabeled_1",
            occurred_at=occurred_at,
            received_at=occurred_at,
            text="불안해",
            locale="ko",
            source_type="search",
            surface_type="search_box",
        )
    )
    stored = captured_repo.get("unlabeled_1")
    assert stored is not None
    captured_repo.save_generated_view(
        CapturedTextGeneratedViewRecord(
            event_id="unlabeled_1",
            generated_at=occurred_at,
            weak_text="I am anxious",
            strong_text_0="I feel anxious now",
            strong_text_1="I am worried now",
            generator_name="unit-test",
            generator_version="v1",
            source_text_fingerprint=stored.text_fingerprint,
            metadata={"weak_text_translated": True},
        )
    )
    captured_repo.mark_view_generation_status(
        event_id="unlabeled_1",
        status=CAPTURED_TEXT_VIEW_STATUS_READY,
    )
    active_state = make_peft_classifier_state_payload(
        model_id="tracemind-embed",
        model_revision="rev_001",
        training_scope="adapter_only",
        backbone=peft_config.PeftEncoderTrainingBackendConfig().to_backbone_payload(),
        peft_adapter_config=(
            peft_config.PeftEncoderTrainingBackendConfig().to_peft_adapter_config_payload()
        ),
        label_schema=("anxiety", "normal"),
        updated_at=occurred_at,
    )
    update_payload = make_peft_classifier_delta_payload(
        model_id="tracemind-embed",
        base_model_revision="rev_001",
        training_scope="adapter_only",
        backbone=peft_config.PeftEncoderTrainingBackendConfig().to_backbone_payload(),
        peft_adapter_config=(
            peft_config.PeftEncoderTrainingBackendConfig().to_peft_adapter_config_payload()
        ),
        label_schema=("anxiety", "normal"),
        example_count=1,
        peft_parameter_deltas={"lora.test": [0.1]},
        classifier_head_weight_deltas={"anxiety": [0.1], "normal": [-0.1]},
        classifier_head_bias_deltas={"anxiety": 0.01, "normal": -0.01},
        delta_format="inline_delta",
    )
    captured_call: dict[str, object] = {}
    usage_ledger = TrainingUsageLedgerRepository(db_path=tmp_path / "usage.db")

    def _method_core(**kwargs: object) -> QuerySslPeftEncoderClientTrainingResult:
        captured_call.update(kwargs)
        training_task = kwargs["training_task"]
        model_manifest = kwargs["model_manifest"]
        update_envelope = make_training_update_envelope(
            update_id="update_fedmatch_test",
            round_id=training_task.round_id,
            task_id=training_task.task_id,
            model_id=model_manifest.model_id,
            base_model_revision=model_manifest.model_revision,
            training_scope=training_task.training_scope,
            payload_ref="client-submission::update_fedmatch_test",
            payload_format=PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
            example_count=update_payload.example_count,
            client_metrics={"fedmatch_local_runtime": 1.0},
        )
        return QuerySslPeftEncoderClientTrainingResult(
            update_envelope=update_envelope,
            update_payload=update_payload,
            candidate_count=2,
            accepted_count=1,
            local_step_plan=build_query_ssl_local_step_plan(
                labeled_loader_steps=1,
                unlabeled_loader_steps=1,
                uses_labeled_batches=True,
                local_epochs=1,
                max_steps=1,
            ),
            client_metrics=update_envelope.client_metrics,
        )

    round_client = MagicMock()
    service = AgentQuerySslTrainingTaskService(
        method_owned_training_core=_method_core,
        usage_ledger_repository=usage_ledger,
    )

    result = service.run_current_task(
        AgentQuerySslTrainingTaskRunRequest(
            training_task=_query_ssl_task(
                fssl_method="fedmatch",
                fssl_context={
                    "schema_version": "fssl_context.v1",
                    "method_name": "fedmatch",
                    "context_kind": "peer_context",
                    "peer_context": {
                        "schema_version": "peer_context_task.v1",
                        "policy_name": "previous_round_metric_summary",
                        "source_round_id": "round_prev",
                        "warmup": False,
                        "client_contexts": [
                            {
                                "client_id": "agent_01",
                                "helper_client_ids": ["helper_02"],
                            }
                        ],
                    },
                },
            ),
            model_manifest=make_embedding_manifest(
                model_id="tracemind-embed",
                model_revision="rev_001",
                training_scope="adapter_only",
                artifact_ref="/tmp/rev_001.json",
            ),
            active_state=active_state,
            round_client=round_client,
            analysis_event_repository=scored_repo,
            captured_text_repository=captured_repo,
            analysis_event_days=7,
            agent_id="agent_01",
        )
    )

    assert result.status == "uploaded"
    assert result.update_id == "update_fedmatch_test"
    assert captured_call["local_ssl_policy_name"] == "fedmatch_agreement"
    assert captured_call["ssl_method_config"].name == "fedmatch"
    assert captured_call["ssl_method_config"].scenario == "labels-at-client"
    assert captured_call["peer_context"].helper_client_ids == ("helper_02",)
    usage_run = usage_ledger.get_run("update_fedmatch_test")
    assert usage_run is not None
    assert usage_run.objective_method_name == "fedmatch"
    assert usage_run.objective_algorithm_name == "federated_ssl"
    assert usage_run.metadata["effective_method_family"] == "federated_ssl"
    assert usage_run.metadata["fssl_method"] == "fedmatch"
    assert usage_run.metadata["query_ssl_method_name"] == "fixmatch_usb_v1"
    assert usage_run.metadata["local_epochs"] == 1
    assert usage_run.metadata["max_steps"] == 2
    round_client.upload_update.assert_called_once()


def _query_ssl_task(
    *,
    fssl_method: str | None = None,
    fssl_context: dict[str, object] | None = None,
) -> TrainingTaskPayload:
    return TrainingTaskPayload(
        schema_version="training_task.v1",
        task_id="task_query_ssl",
        round_id="round_001",
        model_id="tracemind-embed",
        model_revision="rev_001",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        max_steps=2,
        objective_config=TrainingObjectiveConfigPayload(
            algorithm_profile_name="peft_classifier_update_v1",
            training_backend_name="peft_classifier_trainer",
            privacy_guard_name="noop",
            extras={
                "query_ssl.method_name": "fixmatch_usb_v1",
                "query_ssl.algorithm_name": "fixmatch",
                "query_ssl.strong_view_policy": "first_aug",
                "query_ssl.unlabeled_batch_size": 4,
                "query_ssl.temperature": 0.5,
                "query_ssl.p_cutoff": 0.95,
                "query_ssl.hard_label": True,
                "query_ssl.lambda_u": 1.0,
                "query_ssl.supervised_loss_weight": 1.0,
                "peft_classifier.delta_format": "inline_delta",
            },
        ),
        selection_policy=TrainingSelectionPolicyPayload(max_examples=10),
        fssl_method=fssl_method,
        fssl_context=fssl_context,
    )
