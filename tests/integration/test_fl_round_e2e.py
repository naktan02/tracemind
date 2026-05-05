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

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.src.services.federation.rounds.round_client import RoundClient
from agent.src.services.federation.rounds.runtime_service import (
    FederationRunStatus,
    FederationRuntimeService,
)
from main_server.src.api.fl_rounds import get_round_lifecycle_service
from main_server.src.api.main import app as server_app
from main_server.src.infrastructure.repositories import (
    model_manifest_repository as model_manifest_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_update_repository as shared_adapter_update_repository_module,
)
from main_server.src.infrastructure.repositories.round_repository import RoundRepository
from main_server.src.services.federation.rounds.active_manifest_service import (
    ActiveModelManifestService,
)
from main_server.src.services.federation.rounds.round_lifecycle_service import (
    RoundLifecycleService,
)
from shared.src.contracts.adapter_contracts import (
    dump_vector_adapter_state_payload,
    make_diagonal_delta_payload,
    make_identity_state_payload,
)
from shared.src.contracts.model_contracts import (
    ModelManifest,
    make_embedding_manifest,
)
from shared.src.contracts.training_contracts import (
    make_training_update_envelope,
    make_training_update_submission,
)

# ------------------------------------------------------------------ #
# 픽스처                                                               #
# ------------------------------------------------------------------ #

MODEL_ID = "test-model"
MODEL_REVISION = "rev_test_001"
EMBEDDING_DIM = 8  # 테스트용 소형 차원
SharedAdapterUpdateRepository = (
    shared_adapter_update_repository_module.SharedAdapterUpdateRepository
)
ModelManifestRepository = model_manifest_repository_module.ModelManifestRepository


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
    """identity adapter state 파일을 생성하고 경로를 반환한다."""
    path = artifact_root / "base_state.json"
    payload = make_identity_state_payload(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_dim=EMBEDDING_DIM,
    )
    dump_vector_adapter_state_payload(path, payload)
    return path


@pytest.fixture()
def delta_payload():
    """서버에 inline 제출할 diagonal delta payload."""
    return make_diagonal_delta_payload(
        model_id=MODEL_ID,
        base_model_revision=MODEL_REVISION,
        dimension_deltas=[0.01] * EMBEDDING_DIM,
        example_count=5,
        mean_confidence=0.82,
    )


@pytest.fixture()
def round_service(state_root: Path, artifact_root: Path) -> RoundLifecycleService:
    """격리된 state_root를 사용하는 RoundLifecycleService."""
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


def _active_manifest_payload(base_state_path: Path) -> dict:
    """active manifest 등록 요청 payload dict."""
    manifest = make_embedding_manifest(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        prototype_version="proto_test",
        artifact_ref=str(base_state_path),
    )
    return manifest.model_dump(mode="json")


def _open_round_payload() -> dict:
    """round open 요청 payload dict."""
    return {}


def _activate_manifest(
    server_client: httpx.Client,
    base_state_path: Path,
) -> dict:
    """서버 active manifest를 bootstrap한다."""
    response = server_client.post(
        "/api/v1/fl/rounds/active-manifest",
        json=_active_manifest_payload(base_state_path),
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
        prototype_version="proto_test",
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
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
    base_state_path: Path,
) -> None:
    """round open API가 open 상태 record를 반환한다."""
    _activate_manifest(server_client, base_state_path)
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
    base_state_path: Path,
) -> None:
    """RoundClient가 open round의 training task를 가져온다."""
    _activate_manifest(server_client, base_state_path)
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
) -> None:
    """round가 열려 있어도 예시 없으면 INSUFFICIENT_EXAMPLES를 반환한다."""
    _activate_manifest(server_client, base_state_path)
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
    base_state_path: Path,
    artifact_root: Path,
    delta_payload,
) -> None:
    """round 생성 → update 업로드 → finalize 전체 흐름이 정상 작동한다."""
    _activate_manifest(server_client, base_state_path)
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
        example_count=5,
        client_metrics={"mean_confidence": 0.82, "mean_margin": 0.14},
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
    accepted_payload_ref = Path(accepted_update.payload_ref)
    assert accepted_payload_ref.exists()
    assert accepted_payload_ref.is_relative_to(artifact_root / "server_updates")
    assert accepted_payload_ref != Path(envelope.payload_ref)

    # 3. finalize
    r = server_client.post(
        f"/api/v1/fl/rounds/{round_id}/finalize",
        json={"next_prototype_version": "proto_test_v2"},
    )
    assert r.status_code == 200
    final = r.json()
    assert final["status"] == "finalized"
    assert final["publication"] is not None
    assert final["publication"]["update_count"] == 1
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
