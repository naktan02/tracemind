"""Agent active training task 실행 application service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRepository,
)
from agent.src.infrastructure.repositories.scored_event_repository import (
    ScoredEventRepository,
)
from agent.src.services.assets.adapters.composition_service import (
    AdapterCompositionService,
)
from agent.src.services.assets.prototypes.runtime_service import PrototypeRuntimeService
from agent.src.services.assets.prototypes.sync_service import PrototypeSyncService
from agent.src.services.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from agent.src.services.assets.shared_adapters.sync_service import (
    SharedAdapterSyncService,
)
from agent.src.services.federation.rounds.round_client import RoundClient
from agent.src.services.federation.rounds.runtime_service import (
    FederationRunResult,
    FederationRuntimeService,
)
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.training.backends.inputs.models import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
    TrainingExampleSource,
)
from agent.src.services.training.datasets.captured_text_training_source_service import (
    CapturedTextTrainingSourceService,
)
from agent.src.services.training.examples.service import TrainingExampleService
from agent.src.services.training.execution.runtime_compatibility import (
    validate_local_training_runtime,
)
from shared.src.contracts.model_contracts import PROTOTYPE_PACK_AUXILIARY_KEY
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter

RoundClientFactory = Callable[[str], RoundClient]
FederationRuntimeServiceFactory = Callable[[str], FederationRuntimeService]


@dataclass(slots=True, frozen=True)
class AgentTrainingTaskRunRequest:
    """Agent current task 실행 입력."""

    server_base_url: str
    scored_event_days: int = 7
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

    scored_event_repository: ScoredEventRepository
    prototype_runtime_service: PrototypeRuntimeService
    prototype_sync_service: PrototypeSyncService
    shared_adapter_runtime_service: SharedAdapterRuntimeService
    shared_adapter_sync_service: SharedAdapterSyncService
    round_client_factory: RoundClientFactory
    federation_runtime_service_factory: FederationRuntimeServiceFactory
    captured_text_repository: CapturedTextRepository | None = None
    embedding_adapter: EmbeddingAdapter | None = None

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

        prototype_version = active_manifest.auxiliary_artifact_versions.get(
            PROTOTYPE_PACK_AUXILIARY_KEY
        )
        if prototype_version is None:
            training_examples = ()
        else:
            try:
                self.prototype_sync_service.pull_version(
                    server_base_url=request.server_base_url,
                    prototype_version=prototype_version,
                )
                active_pack = self.prototype_runtime_service.get_active_pack()
            except FileNotFoundError:
                training_examples = ()
            else:
                scoring_service = ScoringService.from_objective_config(
                    task_payload.objective_config,
                    shared_state=active_state,
                )
                training_example_service = TrainingExampleService.from_objective_config(
                    task_payload.objective_config
                )
                if training_example_service.backend.supports_stored_event_rebuild:
                    stored_events = self.scored_event_repository.get_recent_stored(
                        days=request.scored_event_days
                    )
                    training_examples = (
                        training_example_service.build_examples_from_stored_events(
                            StoredEventTrainingExampleBuildRequest(
                                stored_events=stored_events,
                                prototype_pack=active_pack,
                                scoring_service=scoring_service,
                                adapter_state=active_state,
                            )
                        )
                    )
                else:
                    source_rows = self._load_captured_text_source_rows(
                        days=request.scored_event_days,
                        max_examples=task_payload.selection_policy.max_examples,
                    )
                    if source_rows and self.embedding_adapter is None:
                        return AgentTrainingTaskRunResult(
                            status="missing_embedding_adapter",
                            round_id=task_payload.round_id,
                            task_id=task_payload.task_id,
                            message=(
                                "generated captured text source를 학습에 쓰려면 "
                                "embedding_adapter가 필요합니다."
                            ),
                        )
                    if not source_rows or self.embedding_adapter is None:
                        training_examples = ()
                    else:
                        training_examples = training_example_service.build_examples(
                            TrainingExampleBuildRequest(
                                source_rows=source_rows,
                                adapter=self.embedding_adapter,
                                adapter_state=active_state,
                                prototype_pack=active_pack,
                                model_id=task_payload.model_id,
                                scoring_service=scoring_service,
                            )
                        )

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

    def _load_captured_text_source_rows(
        self,
        *,
        days: int,
        max_examples: int | None,
    ) -> tuple[TrainingExampleSource, ...]:
        if self.captured_text_repository is None:
            return ()
        service = CapturedTextTrainingSourceService(
            repository=self.captured_text_repository
        )
        return service.get_recent_source_rows(
            days=days,
            limit=max_examples or 100,
        )
