"""Agent active training task 실행 application service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRepository,
)
from agent.src.services.assets.adapters.composition_service import (
    AdapterCompositionService,
)
from agent.src.services.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from agent.src.services.assets.shared_adapters.sync_service import (
    SharedAdapterSyncService,
)
from agent.src.services.federation.rounds.artifact_client import RoundArtifactClient
from agent.src.services.federation.rounds.round_client import RoundClient
from agent.src.services.federation.rounds.runtime_service import (
    FederationRunResult,
    FederationRuntimeService,
)
from agent.src.services.training.execution.query_ssl_training_task_service import (
    AgentQuerySslTrainingTaskRunRequest,
    AgentQuerySslTrainingTaskService,
)
from agent.src.services.training.execution.runtime_compatibility import (
    validate_local_training_runtime,
)
from methods.ssl.runtime.objective_config import QuerySslObjectiveRuntimeConfig

RoundClientFactory = Callable[[str], RoundClient]
RoundArtifactClientFactory = Callable[[str], RoundArtifactClient]
FederationRuntimeServiceFactory = Callable[[str], FederationRuntimeService]


@dataclass(slots=True, frozen=True)
class AgentTrainingTaskRunRequest:
    """Agent current task 실행 입력."""

    server_base_url: str
    analysis_event_days: int = 7
    agent_id: str | None = None


@dataclass(slots=True, frozen=True)
class AgentTrainingTaskRunResult:
    """Agent current task 실행 결과."""

    status: str
    round_id: str | None = None
    task_id: str | None = None
    update_id: str | None = None
    example_count: int = 0
    accepted_count: int = 0
    message: str = ""


@dataclass(slots=True)
class AgentTrainingTaskRunnerService:
    """agent current-task application flow를 소유한다.

    fetch/sync/example build/runtime upload를 route 대신 실행한다.
    """

    analysis_event_repository: AnalysisEventRepository
    shared_adapter_runtime_service: SharedAdapterRuntimeService
    shared_adapter_sync_service: SharedAdapterSyncService
    round_client_factory: RoundClientFactory
    federation_runtime_service_factory: FederationRuntimeServiceFactory
    round_artifact_client_factory: RoundArtifactClientFactory = RoundArtifactClient
    captured_text_repository: CapturedTextRepository | None = None
    query_ssl_task_service: AgentQuerySslTrainingTaskService = field(
        default_factory=AgentQuerySslTrainingTaskService
    )

    def run_current_task(
        self,
        request: AgentTrainingTaskRunRequest,
    ) -> AgentTrainingTaskRunResult:
        """현재 active task를 읽어 로컬 학습을 실행하고 update를 업로드한다."""

        round_client = self.round_client_factory(request.server_base_url)
        task_payload = round_client.fetch_current_task()
        if task_payload is None:
            return AgentTrainingTaskRunResult(
                status="no_active_task",
                message="현재 active round 또는 open task가 없습니다.",
            )
        query_ssl_config = QuerySslObjectiveRuntimeConfig.from_objective_config(
            task_payload.objective_config
        )

        if query_ssl_config is None:
            try:
                validate_local_training_runtime(task_payload)
            except ValueError as error:
                return AgentTrainingTaskRunResult(
                    status="unsupported_runtime",
                    round_id=task_payload.round_id,
                    task_id=task_payload.task_id,
                    message=str(error),
                )

        try:
            self.shared_adapter_sync_service.pull_current(
                server_base_url=request.server_base_url
            )
            adapter_context = AdapterCompositionService(
                shared_adapter_provider=self.shared_adapter_runtime_service,
            ).get_context(require_shared=True)
            active_manifest = adapter_context.require_shared_manifest()
            active_state = adapter_context.require_shared_state()
        except FileNotFoundError as error:
            return AgentTrainingTaskRunResult(
                status="no_active_shared_state",
                round_id=task_payload.round_id,
                task_id=task_payload.task_id,
                message=str(error),
            )

        if active_manifest.model_revision != task_payload.model_revision:
            return AgentTrainingTaskRunResult(
                status="stale_shared_state",
                round_id=task_payload.round_id,
                task_id=task_payload.task_id,
                message=(
                    "로컬 shared adapter revision이 task revision과 다릅니다: "
                    f"{active_manifest.model_revision} != "
                    f"{task_payload.model_revision}"
                ),
            )

        if query_ssl_config is not None:
            try:
                result = self.query_ssl_task_service.run_current_task(
                    AgentQuerySslTrainingTaskRunRequest(
                        training_task=task_payload,
                        model_manifest=active_manifest,
                        active_state=active_state,
                        round_client=round_client,
                        artifact_loader=self.round_artifact_client_factory(
                            request.server_base_url
                        ),
                        analysis_event_repository=self.analysis_event_repository,
                        captured_text_repository=self.captured_text_repository,
                        analysis_event_days=request.analysis_event_days,
                        agent_id=request.agent_id,
                    )
                )
            except ValueError as error:
                return AgentTrainingTaskRunResult(
                    status="unsupported_runtime",
                    round_id=task_payload.round_id,
                    task_id=task_payload.task_id,
                    message=str(error),
                )
            return AgentTrainingTaskRunResult(
                status=str(result.status),
                round_id=result.round_id,
                task_id=result.task_id,
                update_id=result.update_id,
                example_count=result.example_count,
                accepted_count=result.accepted_count,
                message=result.message,
            )

        training_examples = ()

        service = self.federation_runtime_service_factory(request.server_base_url)
        result: FederationRunResult = service.run_current_task(
            training_examples=training_examples,
            model_manifest=active_manifest,
            agent_id=request.agent_id,
            task_payload=task_payload,
        )
        return AgentTrainingTaskRunResult(
            status=str(result.status),
            round_id=result.round_id,
            task_id=result.task_id,
            update_id=result.update_id,
            example_count=result.example_count,
            accepted_count=result.accepted_count,
            message=result.message,
        )
