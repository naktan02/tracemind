"""Agent Query SSL training task service tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from agent.src.infrastructure.repositories.captured_text_repository import (
    CAPTURED_TEXT_VIEW_STATUS_READY,
    CapturedTextGeneratedViewRecord,
    CapturedTextRecord,
    CapturedTextRepository,
)
from agent.src.infrastructure.repositories.scored_event_repository import (
    ScoredEventRepository,
)
from agent.src.services.training.execution.query_ssl_training_task_service import (
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
from shared.src.domain.entities.inference.events import ScoredEvent

QuerySslPeftEncoderClientTrainingResult = (
    qssl_training.QuerySslPeftEncoderClientTrainingResult
)


class _QuerySslPeftBackend:
    backend_name = "peft_classifier_trainer"

    def __init__(self, update_payload: PeftClassifierDelta) -> None:
        self.update_payload = update_payload
        self.captured_kwargs: dict[str, object] | None = None

    def matches_objective_config(self, objective_config: object | None) -> bool:
        del objective_config
        return True

    def build_query_ssl_update(
        self,
        **kwargs: object,
    ) -> QuerySslPeftEncoderClientTrainingResult:
        self.captured_kwargs = dict(kwargs)
        training_task = kwargs["training_task"]
        model_manifest = kwargs["model_manifest"]
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
    scored_repo = ScoredEventRepository(db_path=tmp_path / "scored.db")
    scored_repo.save(
        ScoredEvent(
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
    service = AgentQuerySslTrainingTaskService(backend=backend)

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
            scored_event_repository=scored_repo,
            captured_text_repository=captured_repo,
            scored_event_days=7,
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


def _query_ssl_task() -> TrainingTaskPayload:
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
    )
