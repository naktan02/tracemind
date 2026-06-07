"""FL round E2E HTTP 통합 테스트.

httpx ASGITransport을 사용해 실제 소켓/포트 없이 in-process로 서버를 테스트한다.
FastAPI dependency_overrides로 RoundLifecycleService를 격리된 인스턴스로 교체한다.

실행 방법:
  uv run pytest tests/integration/ -v -m integration

테스트 범위:
  1. 서버 health 확인
  2. round 생성 + 상태 확인
  3. round 없을 때 NO_ACTIVE_TASK (RoundClient 경유)
  4. 예시 없을 때 INSUFFICIENT_EXAMPLES
  5. round 생성 → update 업로드 → finalize 전체 흐름
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from agent.src.infrastructure.repositories.captured_text_repository import (
    CAPTURED_TEXT_VIEW_STATUS_READY,
    CapturedTextGeneratedViewRecord,
    CapturedTextRecord,
    CapturedTextRepository,
)
from agent.src.services.federation.rounds.round_client import RoundClient
from agent.src.services.federation.rounds.runtime_service import (
    FederationRunResult,
    FederationRunStatus,
    FederationRuntimeService,
)
from agent.src.services.training.execution.agent_training_task_runner_service import (
    AgentTrainingTaskRunnerService,
    AgentTrainingTaskRunRequest,
)
from agent.src.services.training.execution.query_ssl_training_task_service import (
    AgentQuerySslTrainingTaskRunRequest,
    AgentQuerySslTrainingTaskService,
)
from main_server.src.api.fl_rounds import get_round_lifecycle_service
from main_server.src.api.main import app as server_app
from main_server.src.infrastructure.repositories import (
    model_manifest_repository as model_manifest_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as shared_adapter_state_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_update_repository as shared_adapter_update_repository_module,
)
from main_server.src.infrastructure.repositories.round_repository import RoundRepository
from main_server.src.services.federation.rounds.active_manifest_service import (
    ActiveModelManifestService,
)
from main_server.src.services.federation.rounds.payload_adapters.registry import (
    build_shared_adapter_round_payload_adapter,
)
from main_server.src.services.federation.rounds.round_lifecycle_service import (
    RoundLifecycleService,
)
from main_server.src.services.federation.rounds.round_manager_service import (
    RoundManagerService,
)
from methods.adaptation.peft_text_encoder.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.query_text_views.local_training_budget import (
    build_query_ssl_local_step_plan,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    CLASSIFIER_HEAD_UPDATE_PAYLOAD_FORMAT,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_classifier_head_delta_payload,
    make_peft_classifier_delta_payload,
    make_peft_classifier_state_payload,
    make_zero_classifier_head_state_payload,
)
from shared.src.contracts.adapter_contract_families.io import (
    dump_shared_adapter_state_payload,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    PeftClassifierDelta,
)
from shared.src.contracts.common_types import TrainingScope
from shared.src.contracts.model_contracts import (
    ModelManifest,
    make_embedding_manifest,
)
from shared.src.contracts.training_contracts import (
    make_training_update_envelope,
    make_training_update_submission,
)
from shared.src.domain.entities.inference.events import AnalysisEvent

# ------------------------------------------------------------------ #
# 픽스처                                                               #
# ------------------------------------------------------------------ #

MODEL_ID = "test-model"
MODEL_REVISION = "rev_test_001"
EMBEDDING_DIM = 8  # 테스트용 소형 차원
LABELS = ("negative", "positive")
SharedAdapterUpdateRepository = (
    shared_adapter_update_repository_module.SharedAdapterUpdateRepository
)
SharedAdapterStateRepository = (
    shared_adapter_state_repository_module.SharedAdapterStateRepository
)
ModelManifestRepository = model_manifest_repository_module.ModelManifestRepository
QuerySslPeftEncoderClientTrainingResult = (
    qssl_training.QuerySslPeftEncoderClientTrainingResult
)


@pytest.fixture()
def state_root(tmp_path: Path) -> Path:
    """격리된 round state 디렉토리."""
    return tmp_path / "rounds"


@pytest.fixture()
def artifact_root(tmp_path: Path) -> Path:
    """격리된 artifact 파일 디렉토리."""
    root = tmp_path / "artifacts"
    root.mkdir(parents=True)
    return root


@pytest.fixture()
def base_state_path(artifact_root: Path) -> Path:
    """classifier-head adapter state 파일을 생성하고 경로를 반환한다."""
    path = artifact_root / "base_state.json"
    payload = make_zero_classifier_head_state_payload(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        labels=LABELS,
        embedding_dim=EMBEDDING_DIM,
    )
    dump_shared_adapter_state_payload(path, payload)
    return path


@pytest.fixture()
def delta_payload():
    """서버에 inline 제출할 classifier-head delta payload."""
    return make_classifier_head_delta_payload(
        model_id=MODEL_ID,
        base_model_revision=MODEL_REVISION,
        label_weight_deltas={
            "negative": [0.01] * EMBEDDING_DIM,
            "positive": [-0.01] * EMBEDDING_DIM,
        },
        label_bias_deltas={"negative": 0.01, "positive": -0.01},
        example_count=5,
        mean_confidence=0.82,
        mean_margin=0.14,
    )


@pytest.fixture()
def round_service(state_root: Path, artifact_root: Path) -> RoundLifecycleService:
    """격리된 state_root를 사용하는 RoundLifecycleService."""
    state_repository = SharedAdapterStateRepository(
        state_root=artifact_root / "shared_states"
    )
    return RoundLifecycleService(
        round_repository=RoundRepository(state_root=state_root),
        update_payload_repository=SharedAdapterUpdateRepository(
            state_root=artifact_root / "server_updates"
        ),
        active_manifest_service=ActiveModelManifestService(
            manifest_repository=ModelManifestRepository(
                state_root=artifact_root / "model_manifests"
            )
        ),
        round_manager_service=RoundManagerService(
            payload_adapter=build_shared_adapter_round_payload_adapter(
                "classifier_head",
                aggregation_backend_name="fedavg",
            ),
            artifact_repository=state_repository,
        ),
    )


@pytest.fixture()
def server_client(round_service: RoundLifecycleService) -> TestClient:
    """TestClient로 main_server와 통신한다.

    dependency_overrides로 RoundLifecycleService를 격리된 인스턴스로 교체한다.
    """
    server_app.dependency_overrides[get_round_lifecycle_service] = lambda: round_service
    with TestClient(server_app, base_url="http://testserver") as client:
        yield client
    server_app.dependency_overrides.clear()


class _BorrowedClientContext(AbstractContextManager[httpx.Client]):
    """이미 열린 TestClient를 RoundClient가 재사용하도록 감싸는 context."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def __enter__(self) -> httpx.Client:
        return self._client

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _QuerySslUploadSmokeBackend:
    """Query SSL upload smoke에서 외부 모델 학습만 대체하는 fake backend."""

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
            update_id="update_query_ssl_upload_smoke",
            round_id=training_task.round_id,
            task_id=training_task.task_id,
            model_id=model_manifest.model_id,
            base_model_revision=model_manifest.model_revision,
            training_scope=training_task.training_scope,
            payload_ref="client-submission::update_query_ssl_upload_smoke",
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


@pytest.fixture()
def round_client(server_client: TestClient) -> RoundClient:
    """열린 TestClient를 재사용하는 RoundClient."""
    return RoundClient(
        server_base_url="http://testserver",
        _client_factory=lambda: _BorrowedClientContext(server_client),
    )


# ------------------------------------------------------------------ #
# 헬퍼                                                                 #
# ------------------------------------------------------------------ #


def _active_manifest_payload(round_service: RoundLifecycleService) -> dict:
    """active manifest 등록 요청 payload dict."""
    state_repository = round_service.round_manager_service.artifact_repository
    state_repository.save_shared_adapter_state(
        make_zero_classifier_head_state_payload(
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            labels=LABELS,
            embedding_dim=EMBEDDING_DIM,
        )
    )
    manifest = make_embedding_manifest(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        auxiliary_artifact_versions={"prototype_pack": "proto_test"},
        artifact_ref=state_repository.ref_for_revision(MODEL_REVISION),
        training_scope=TrainingScope.HEAD_ONLY,
    )
    return manifest.model_dump(mode="json")


def _peft_active_manifest_payload(round_service: RoundLifecycleService) -> dict:
    """PEFT shared adapter active manifest 등록 요청 payload dict."""
    state_repository = round_service.round_manager_service.artifact_repository
    state_repository.save_shared_adapter_state(
        _peft_state_payload(model_revision=MODEL_REVISION)
    )
    manifest = make_embedding_manifest(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        artifact_ref=state_repository.ref_for_revision(MODEL_REVISION),
        training_scope=TrainingScope.ADAPTER_ONLY,
    )
    return manifest.model_dump(mode="json")


def _peft_state_payload(*, model_revision: str):
    return make_peft_classifier_state_payload(
        model_id=MODEL_ID,
        model_revision=model_revision,
        training_scope=TrainingScope.ADAPTER_ONLY,
        backbone={
            "backbone_model_id": "mxbai",
            "backbone_revision": "main",
            "tokenizer_model_id": "mxbai",
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
        label_schema=("anxiety", "normal"),
    )


def _open_round_payload() -> dict:
    """round open 요청 payload dict."""
    return {}


def _activate_manifest(
    server_client: httpx.Client,
    round_service: RoundLifecycleService,
) -> dict:
    """서버 active manifest를 bootstrap한다."""
    response = server_client.post(
        "/api/v1/fl/rounds/active-manifest",
        json=_active_manifest_payload(round_service),
    )
    assert response.status_code == 201
    return response.json()


def _activate_peft_manifest(
    server_client: httpx.Client,
    round_service: RoundLifecycleService,
) -> dict:
    """서버 active PEFT manifest를 bootstrap한다."""
    response = server_client.post(
        "/api/v1/fl/rounds/active-manifest",
        json=_peft_active_manifest_payload(round_service),
    )
    assert response.status_code == 201
    return response.json()


def _make_domain_manifest(base_state_path: Path) -> ModelManifest:
    """RoundClient/FederationRuntimeService에 넘기는 도메인 ModelManifest."""
    return ModelManifest(
        schema_version="model_manifest.v1",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        published_at=datetime.now(tz=timezone.utc),
        artifact_kind="embedding",
        artifact_ref=str(base_state_path),
        auxiliary_artifact_versions={"prototype_pack": "proto_test"},
        training_scope=TrainingScope.HEAD_ONLY,
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
    )


def _build_peft_round_service(
    *,
    state_root: Path,
    artifact_root: Path,
) -> RoundLifecycleService:
    state_repository = SharedAdapterStateRepository(
        state_root=artifact_root / "peft_shared_states"
    )
    return RoundLifecycleService(
        round_repository=RoundRepository(state_root=state_root / "peft_rounds"),
        update_payload_repository=SharedAdapterUpdateRepository(
            state_root=artifact_root / "peft_server_updates"
        ),
        active_manifest_service=ActiveModelManifestService(
            manifest_repository=ModelManifestRepository(
                state_root=artifact_root / "peft_model_manifests"
            )
        ),
        round_manager_service=RoundManagerService(
            payload_adapter=build_shared_adapter_round_payload_adapter(
                "peft_classifier",
                aggregation_backend_name="fedavg",
            ),
            artifact_repository=state_repository,
        ),
    )


# ------------------------------------------------------------------ #
# 통합 테스트                                                           #
# ------------------------------------------------------------------ #


@pytest.mark.integration
def test_server_health(server_client: httpx.Client) -> None:
    """서버 health 엔드포인트가 200을 반환한다."""
    r = server_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.integration
def test_round_open_returns_open_status(
    server_client: httpx.Client,
    round_service: RoundLifecycleService,
) -> None:
    """round open API가 open 상태 record를 반환한다."""
    _activate_manifest(server_client, round_service)
    r = server_client.post(
        "/api/v1/fl/rounds",
        json=_open_round_payload(),
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "open"
    assert "round_id" in body
    assert "training_task" in body


@pytest.mark.integration
def test_no_open_round_returns_404(server_client: httpx.Client) -> None:
    """active round 없을 때 /current가 404를 반환한다."""
    r = server_client.get("/api/v1/fl/rounds/current")
    assert r.status_code == 404


@pytest.mark.integration
def test_round_open_requires_active_manifest(server_client: httpx.Client) -> None:
    """서버 active manifest가 없으면 round를 열 수 없다."""
    r = server_client.post("/api/v1/fl/rounds", json=_open_round_payload())
    assert r.status_code == 404


@pytest.mark.integration
def test_agent_fetches_task_from_open_round(
    round_client: RoundClient,
    server_client: httpx.Client,
    round_service: RoundLifecycleService,
) -> None:
    """RoundClient가 open round의 training task를 가져온다."""
    _activate_manifest(server_client, round_service)
    r = server_client.post(
        "/api/v1/fl/rounds",
        json=_open_round_payload(),
    )
    assert r.status_code == 201
    expected_task_id = r.json()["training_task"]["task_id"]

    task = round_client.fetch_current_task()
    assert task is not None
    assert task.task_id == expected_task_id


@pytest.mark.integration
def test_peft_round_default_query_ssl_task_routes_to_agent_query_ssl_service(
    state_root: Path,
    artifact_root: Path,
) -> None:
    """PEFT no-config round task가 agent Query SSL runner로 분기된다."""
    peft_service = _build_peft_round_service(
        state_root=state_root,
        artifact_root=artifact_root,
    )
    server_app.dependency_overrides[get_round_lifecycle_service] = lambda: peft_service
    try:
        with TestClient(server_app, base_url="http://testserver") as server_client:
            _activate_peft_manifest(server_client, peft_service)
            response = server_client.post("/api/v1/fl/rounds", json={})
            assert response.status_code == 201

            round_client = RoundClient(
                server_base_url="http://testserver",
                _client_factory=lambda: _BorrowedClientContext(server_client),
            )
            task = round_client.fetch_current_task()
            assert task is not None
            assert task.objective_config.extras["query_ssl.algorithm_name"] == (
                "fixmatch"
            )
            assert task.objective_config.extras["peft_classifier.delta_format"] == (
                "inline_delta"
            )

            active_manifest = peft_service.active_manifest_service.get_active_manifest()
            active_state = _peft_state_payload(model_revision=MODEL_REVISION)
            shared_runtime = MagicMock()
            shared_runtime.get_active_manifest.return_value = active_manifest
            shared_runtime.get_active_state.return_value = active_state
            shared_sync = MagicMock()
            query_ssl_task_service = MagicMock()
            query_ssl_task_service.run_current_task.return_value = FederationRunResult(
                status=FederationRunStatus.UPLOADED,
                round_id=task.round_id,
                task_id=task.task_id,
                update_id="update_query_ssl_e2e",
                example_count=2,
                accepted_count=1,
                message="Query SSL update 업로드 완료.",
            )
            runtime_factory = MagicMock()
            runner = AgentTrainingTaskRunnerService(
                analysis_event_repository=MagicMock(),
                shared_adapter_runtime_service=shared_runtime,
                shared_adapter_sync_service=shared_sync,
                round_client_factory=MagicMock(return_value=round_client),
                federation_runtime_service_factory=runtime_factory,
                captured_text_repository=MagicMock(),
                query_ssl_task_service=query_ssl_task_service,
            )

            result = runner.run_current_task(
                AgentTrainingTaskRunRequest(server_base_url="http://testserver")
            )

            assert result.status == str(FederationRunStatus.UPLOADED)
            assert result.update_id == "update_query_ssl_e2e"
            query_ssl_request = query_ssl_task_service.run_current_task.call_args.args[
                0
            ]
            assert query_ssl_request.training_task.task_id == task.task_id
            assert query_ssl_request.model_manifest is active_manifest
            assert query_ssl_request.active_state is active_state
            runtime_factory.assert_not_called()
            shared_sync.pull_current.assert_called_once_with(
                server_base_url="http://testserver"
            )
    finally:
        server_app.dependency_overrides.clear()


@pytest.mark.integration
def test_agent_query_ssl_service_uploads_update_to_main_server(
    state_root: Path,
    artifact_root: Path,
    tmp_path: Path,
) -> None:
    """agent Query SSL service가 main_server에 PEFT update를 업로드한다."""
    peft_service = _build_peft_round_service(
        state_root=state_root,
        artifact_root=artifact_root,
    )
    server_app.dependency_overrides[get_round_lifecycle_service] = lambda: peft_service
    try:
        with TestClient(server_app, base_url="http://testserver") as server_client:
            _activate_peft_manifest(server_client, peft_service)
            response = server_client.post("/api/v1/fl/rounds", json={})
            assert response.status_code == 201

            round_client = RoundClient(
                server_base_url="http://testserver",
                _client_factory=lambda: _BorrowedClientContext(server_client),
            )
            training_task = round_client.fetch_current_task()
            assert training_task is not None

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
                    generator_name="integration-test",
                    generator_version="v1",
                    source_text_fingerprint=stored.text_fingerprint,
                    metadata={"weak_text_translated": True},
                )
            )
            captured_repo.mark_view_generation_status(
                event_id="unlabeled_1",
                status=CAPTURED_TEXT_VIEW_STATUS_READY,
            )

            active_manifest = peft_service.active_manifest_service.get_active_manifest()
            active_state = peft_service.get_current_shared_adapter_state().state
            update_payload = make_peft_classifier_delta_payload(
                model_id=MODEL_ID,
                base_model_revision=MODEL_REVISION,
                training_scope=TrainingScope.ADAPTER_ONLY,
                backbone=active_state.backbone,
                peft_adapter_config=active_state.peft_adapter_config,
                label_schema=active_state.label_schema,
                example_count=1,
                peft_parameter_deltas={"lora.test": [0.1]},
                classifier_head_weight_deltas={
                    "anxiety": [0.1],
                    "normal": [-0.1],
                },
                classifier_head_bias_deltas={
                    "anxiety": 0.01,
                    "normal": -0.01,
                },
                delta_format="inline_delta",
            )
            backend = _QuerySslUploadSmokeBackend(update_payload)
            service = AgentQuerySslTrainingTaskService(backend=backend)

            result = service.run_current_task(
                AgentQuerySslTrainingTaskRunRequest(
                    training_task=training_task,
                    model_manifest=active_manifest,
                    active_state=active_state,
                    round_client=round_client,
                    analysis_event_repository=scored_repo,
                    captured_text_repository=captured_repo,
                    analysis_event_days=7,
                    agent_id="agent_01",
                )
            )

            assert result.status == FederationRunStatus.UPLOADED
            assert result.update_id == "update_query_ssl_upload_smoke"
            assert backend.captured_kwargs is not None
            assert backend.captured_kwargs["labels"] == ("anxiety", "normal")
            stored_round = peft_service.get_round(training_task.round_id)
            assert len(stored_round.updates) == 1
            accepted_update = stored_round.updates[0]
            assert accepted_update.update_id == "update_query_ssl_upload_smoke"
            assert accepted_update.client_metrics == {}
            assert accepted_update.payload_ref == (
                peft_service.update_payload_repository.ref_for_update(
                    accepted_update.update_id
                )
            )
            update_repository = peft_service.update_payload_repository
            stored_update = update_repository.load_shared_adapter_update_from_ref(
                accepted_update.payload_ref
            )
            assert stored_update.model_id == MODEL_ID
            assert stored_update.example_count == 1
            assert stored_update.label_schema == ["anxiety", "normal"]
    finally:
        server_app.dependency_overrides.clear()


@pytest.mark.integration
def test_no_active_task_when_no_open_round(
    round_client: RoundClient,
    base_state_path: Path,
) -> None:
    """round가 없으면 FederationRuntimeService가 NO_ACTIVE_TASK를 반환한다."""
    service = FederationRuntimeService(round_client=round_client)
    result = service.run_current_task(
        training_examples=(),
        model_manifest=_make_domain_manifest(base_state_path),
    )
    assert result.status == FederationRunStatus.NO_ACTIVE_TASK


@pytest.mark.integration
def test_insufficient_examples_when_round_open(
    round_client: RoundClient,
    server_client: httpx.Client,
    base_state_path: Path,
    round_service: RoundLifecycleService,
) -> None:
    """round가 열려 있어도 예시 없으면 INSUFFICIENT_EXAMPLES를 반환한다."""
    _activate_manifest(server_client, round_service)
    server_client.post("/api/v1/fl/rounds", json=_open_round_payload())

    service = FederationRuntimeService(round_client=round_client)
    result = service.run_current_task(
        training_examples=(),
        model_manifest=_make_domain_manifest(base_state_path),
    )
    assert result.status == FederationRunStatus.INSUFFICIENT_EXAMPLES


@pytest.mark.integration
def test_full_round_lifecycle(
    server_client: httpx.Client,
    round_service: RoundLifecycleService,
    artifact_root: Path,
    delta_payload,
) -> None:
    """round 생성 → update 업로드 → finalize 전체 흐름이 정상 작동한다."""
    _activate_manifest(server_client, round_service)
    r = server_client.get("/api/v1/fl/rounds/active-state/current")
    assert r.status_code == 200
    assert r.json()["state"]["model_revision"] == MODEL_REVISION

    # 1. round 생성
    r = server_client.post(
        "/api/v1/fl/rounds",
        json=_open_round_payload(),
    )
    assert r.status_code == 201
    record = r.json()
    round_id = record["round_id"]
    task_id = record["training_task"]["task_id"]

    # 2. update 업로드 — factory로 올바른 payload 구성
    envelope = make_training_update_envelope(
        round_id=round_id,
        task_id=task_id,
        model_id=MODEL_ID,
        base_model_revision=MODEL_REVISION,
        payload_ref=str(artifact_root / "agent_only_delta.json"),
        payload_format=CLASSIFIER_HEAD_UPDATE_PAYLOAD_FORMAT,
        example_count=5,
        client_metrics={"mean_confidence": 0.82, "mean_margin": 0.14},
        training_scope=TrainingScope.HEAD_ONLY,
    )
    submission = make_training_update_submission(
        envelope=envelope,
        update_payload=delta_payload,
    )
    r = server_client.post(
        f"/api/v1/fl/rounds/{round_id}/updates",
        json=submission.model_dump(mode="json"),
    )
    assert r.status_code == 202
    assert r.json()["update_count"] >= 1
    accepted_update = round_service.get_round(round_id).updates[0]
    assert (
        accepted_update.payload_ref
        == round_service.update_payload_repository.ref_for_update(
            accepted_update.update_id
        )
    )
    assert accepted_update.payload_ref != envelope.payload_ref
    round_service.update_payload_repository.load_shared_adapter_update_from_ref(
        accepted_update.payload_ref
    )

    # 3. finalize
    r = server_client.post(
        f"/api/v1/fl/rounds/{round_id}/finalize",
        json={"next_auxiliary_artifact_versions": {"prototype_pack": "proto_test_v2"}},
    )
    assert r.status_code == 200
    final = r.json()
    assert final["status"] == "finalized"
    assert final["publication"] is not None
    assert final["publication"]["update_count"] == 1
    assert final["publication"]["next_manifest"]["artifact_ref"].startswith(
        "shared_adapter_state::"
    )
    next_revision = final["publication"]["next_manifest"]["model_revision"]
    r = server_client.get("/api/v1/fl/rounds/active-manifest/current")
    assert r.status_code == 200
    assert r.json()["model_revision"] == next_revision
    r = server_client.get("/api/v1/fl/rounds/active-state/current")
    assert r.status_code == 200
    assert r.json()["state"]["model_revision"] == next_revision

    r = server_client.post("/api/v1/fl/rounds", json=_open_round_payload())
    assert r.status_code == 201
    assert r.json()["training_task"]["model_revision"] == next_revision
