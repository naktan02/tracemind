"""Federated simulation 핵심 실행 로직."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
    EmbeddingAdapterSpec,
)
from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.federation.training_example_service import (
    TrainingExampleBuildRequest,
    TrainingExampleService,
    TrainingExampleSource,
)
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.training.local_training_service import (
    EmbeddedTrainingExample,
    LocalTrainingRequest,
    LocalTrainingService,
)
from main_server.src.infrastructure.repositories import (
    prototype_build_state_repository,
    prototype_pack_repository,
    prototype_rebuild_input_repository,
    vector_adapter_state_repository,
)
from main_server.src.infrastructure.repositories.round_repository import RoundRepository
from main_server.src.services.prototypes import (
    PrototypeBuildStateService,
    PrototypePackService,
    PrototypeRebuildInputRecord,
    PrototypeRebuildService,
    ReferencePrototypeSourceRow,
    ReferenceRebuildPrototypePublicationStrategy,
    StoredReferencePrototypeRebuildRequest,
    StoredReferencePrototypeRebuildService,
)
from main_server.src.services.rounds.models import (
    RoundFinalizeRequest,
    RoundOpenRequest,
)
from main_server.src.services.rounds.round_lifecycle_service import (
    RoundLifecycleService,
)
from main_server.src.services.rounds.round_manager_service import (
    RoundManagerService,
)
from scripts.experiments.federated_simulation.artifacts import (
    save_model_manifest,
    save_prototype_pack,
    save_selection_diagnostics,
)
from scripts.experiments.federated_simulation.models import (
    ClientRoundSummary,
    FederatedDiagnosticsConfig,
    FederatedPrototypeRebuildConfig,
    FederatedShardPolicyConfig,
    FederatedTrainingTaskConfig,
    FederatedValidationConfig,
    SimulationEvaluation,
    SimulationResult,
    SimulationRoundSummary,
)
from scripts.experiments.federated_simulation.sharding import (
    split_rows_for_federation,
)
from shared.src.contracts.adapter_contracts import DiagonalScaleAdapterStatePayload
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    load_prototype_pack_payload,
)
from shared.src.domain.entities.artifacts.model_manifest import ModelManifest
from shared.src.domain.entities.training.training_task_config import (
    TrainingObjectiveConfig,
)
from shared.src.domain.entities.training.vector_adapter_state import VectorAdapterState
from shared.src.services.prototypes.build_strategies import PrototypeBuildStrategy


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """JSONL 파일을 row 목록으로 읽는다."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class _SimulationEmbeddingAdapterFactory:
    adapter: Any = None

    @classmethod
    def create(cls, spec: EmbeddingAdapterSpec) -> Any:
        del spec
        if cls.adapter is None:
            raise ValueError("Simulation embedding adapter is not configured.")
        return cls.adapter


def _store_prototype_rebuild_input(
    *,
    rows: list[dict[str, Any]],
    embedding_spec: EmbeddingAdapterSpec,
    repository: (
        prototype_rebuild_input_repository.PrototypeRebuildInputRepository
    ),
    rebuild_config: FederatedPrototypeRebuildConfig,
) -> str:
    """bootstrap row를 canonical prototype rebuild input으로 저장한다."""
    input_id = "bootstrap_v1"
    repository.save_input(
        PrototypeRebuildInputRecord(
            input_id=input_id,
            embedding_spec=embedding_spec,
            rows=tuple(
                ReferencePrototypeSourceRow(
                    text=str(row["text"]),
                    category=str(row["mapped_label_4"]),
                )
                for row in rows
            ),
            mapping_version=rebuild_config.mapping_version,
            translation_model_id=rebuild_config.translation_model_id,
            translation_model_revision=rebuild_config.translation_model_revision,
            translation_direction=rebuild_config.translation_direction,
        )
    )
    repository.set_active(input_id)
    return input_id


def _build_prototype_rebuild_runtime_service(
    *,
    output_dir: Path,
    build_strategy: PrototypeBuildStrategy,
    input_repository: (
        prototype_rebuild_input_repository.PrototypeRebuildInputRepository
    ),
) -> StoredReferencePrototypeRebuildService:
    pack_repository = prototype_pack_repository.PrototypePackRepository(
        state_root=output_dir / "main_server" / "runtime_prototype_packs"
    )
    build_state_repository = (
        prototype_build_state_repository.PrototypeBuildStateRepository(
            state_root=output_dir / "main_server" / "runtime_prototype_build_states"
        )
    )
    return StoredReferencePrototypeRebuildService(
        input_repository=input_repository,
        prototype_rebuild_service=PrototypeRebuildService(
            build_strategy=build_strategy,
            publication_strategy=ReferenceRebuildPrototypePublicationStrategy(
                reference_pack_output_dir=(
                    output_dir / "main_server" / "prototype_packs"
                ),
                reference_build_state_output_dir=(
                    output_dir / "main_server" / "prototype_build_states"
                ),
                prototype_pack_service=PrototypePackService(repository=pack_repository),
                prototype_build_state_service=PrototypeBuildStateService(
                    repository=build_state_repository
                ),
            ),
        ),
        adapter_factory=_SimulationEmbeddingAdapterFactory,
    )


def _rebuild_reference_prototype_pack(
    *,
    stored_rebuild_service: StoredReferencePrototypeRebuildService,
    adapter_state: VectorAdapterState,
    prototype_version: str,
    embedding_model_id: str,
    embedding_model_revision: str,
    built_at: datetime,
) -> PrototypePackPayload:
    """reference row 기반 rebuild를 stored runtime service로 실행한다."""
    result = stored_rebuild_service.rebuild(
        StoredReferencePrototypeRebuildRequest(
            adapter_state=adapter_state,
            prototype_version=prototype_version,
            embedding_model_id=embedding_model_id,
            embedding_model_revision=embedding_model_revision,
            built_at=built_at,
        )
    )
    if result.published_pack_path is None:
        raise ValueError("Stored rebuild runtime must publish a prototype pack.")
    return load_prototype_pack_payload(result.published_pack_path)


def _load_active_state(
    *,
    manifest: ModelManifest,
    state_repository: vector_adapter_state_repository.SharedAdapterStateRepository,
    round_manager: RoundManagerService,
) -> VectorAdapterState:
    payload = state_repository.load_state_from_ref(manifest.artifact_ref)
    state = round_manager.adapter_family.state_from_payload(payload)
    if not isinstance(state, VectorAdapterState):
        raise TypeError(
            f"Expected VectorAdapterState from simulation runtime, got {type(state)!r}."
        )
    return state


def build_training_examples(
    *,
    rows: list[dict[str, Any]],
    adapter: Any,
    adapter_state: VectorAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scoring_service: ScoringService,
) -> tuple[EmbeddedTrainingExample, ...]:
    """simulation row를 agent runtime training example builder로 변환한다."""
    service = TrainingExampleService()
    source_rows = tuple(
        TrainingExampleSource(
            query_id=str(row["query_id"]),
            text=str(row["text"]),
            occurred_at=parse_created_at(str(row["created_at"])),
        )
        for row in rows
    )
    return service.build_examples(
        TrainingExampleBuildRequest(
            source_rows=source_rows,
            adapter=adapter,
            adapter_state=adapter_state,
            prototype_pack=prototype_pack,
            model_id=model_id,
            scoring_service=scoring_service,
        )
    )


def evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    adapter: Any,
    adapter_state: VectorAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scoring_service: ScoringService,
    confidence_threshold: float,
    margin_threshold: float,
) -> SimulationEvaluation:
    """validation row에 대해 top1 accuracy와 pseudo-label acceptance 비율을 계산한다."""
    examples = build_training_examples(
        rows=rows,
        adapter=adapter,
        adapter_state=adapter_state,
        prototype_pack=prototype_pack,
        model_id=model_id,
        scoring_service=scoring_service,
    )
    if not examples:
        return SimulationEvaluation(row_count=0, top1_accuracy=0.0, accepted_ratio=0.0)

    correct = 0
    accepted = 0
    for row, example in zip(rows, examples, strict=True):
        ranked_scores = sorted(
            example.scored_event.category_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked_scores[0][0] == str(row["mapped_label_4"]):
            correct += 1
        top_score = ranked_scores[0][1]
        runner_up_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0
        if (
            top_score >= confidence_threshold
            and (top_score - runner_up_score) >= margin_threshold
        ):
            accepted += 1

    return SimulationEvaluation(
        row_count=len(rows),
        top1_accuracy=correct / len(rows),
        accepted_ratio=accepted / len(rows),
    )


def run_simulation(
    *,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    output_dir: Path,
    client_count: int,
    rounds: int,
    bootstrap_ratio: float,
    seed: int,
    embedding_spec: EmbeddingAdapterSpec,
    model_id: str,
    training_scope: str,
    prototype_build_strategy: PrototypeBuildStrategy,
    confidence_threshold: float | None = None,
    margin_threshold: float | None = None,
    max_examples: int | None = None,
    min_required_examples: int | None = None,
    gradient_clip_norm: float | None = None,
    shard_policy: FederatedShardPolicyConfig | None = None,
    training_task_config: FederatedTrainingTaskConfig | None = None,
    validation_config: FederatedValidationConfig | None = None,
    prototype_rebuild_config: FederatedPrototypeRebuildConfig | None = None,
    diagnostics_config: FederatedDiagnosticsConfig | None = None,
) -> SimulationResult:
    """bootstrap -> client pseudo-label -> aggregate -> republish 루프를 실행한다."""
    effective_shard_policy = shard_policy or FederatedShardPolicyConfig()
    effective_training_task_config = training_task_config or _build_legacy_task_config(
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        max_examples=max_examples,
        min_required_examples=min_required_examples,
        gradient_clip_norm=gradient_clip_norm,
    )
    effective_validation_config = validation_config or FederatedValidationConfig(
        confidence_threshold=_resolve_threshold(
            confidence_threshold,
            fallback=effective_training_task_config.objective_config.get(
                "confidence_threshold"
            ),
            default=0.6,
        ),
        margin_threshold=_resolve_threshold(
            margin_threshold,
            fallback=effective_training_task_config.objective_config.get(
                "margin_threshold"
            ),
            default=0.02,
        ),
        score_policy_name=str(
            effective_training_task_config.objective_config.get(
                "score_policy_name",
                "max_cosine",
            )
        ),
        score_top_k=_resolve_optional_positive_int(
            effective_training_task_config.objective_config.get("score_top_k")
        ),
    )
    effective_rebuild_config = (
        prototype_rebuild_config or FederatedPrototypeRebuildConfig()
    )
    effective_diagnostics_config = diagnostics_config or FederatedDiagnosticsConfig()

    dataset_split = split_rows_for_federation(
        train_rows,
        bootstrap_ratio=bootstrap_ratio,
        client_count=client_count,
        seed=seed,
        shard_policy=effective_shard_policy,
    )
    adapter = EmbeddingAdapterFactory.create(embedding_spec)
    if not dataset_split.bootstrap_rows:
        raise ValueError("Bootstrap split must contain at least one row.")
    embedding_dim = len(
        adapter.embed_texts([str(dataset_split.bootstrap_rows[0]["text"])])[0]
    )
    state_repository = vector_adapter_state_repository.SharedAdapterStateRepository(
        state_root=output_dir / "main_server" / "shared_adapter_states"
    )
    round_manager = RoundManagerService(artifact_repository=state_repository)
    round_repository = RoundRepository(state_root=output_dir / "main_server" / "rounds")

    validation_scoring_service = _build_validation_scoring_service(
        effective_validation_config
    )

    initial_model_revision = "sim_rev_0000"
    initial_prototype_version = "proto_sim_0000"
    now = datetime.now(timezone.utc)
    initial_state = VectorAdapterState.identity(
        model_id=model_id,
        model_revision=initial_model_revision,
        training_scope=training_scope,
        embedding_dim=embedding_dim,
        updated_at=now,
    )
    initial_state_path = state_repository.save_shared_adapter_state(
        DiagonalScaleAdapterStatePayload(
            schema_version=initial_state.schema_version,
            adapter_kind=initial_state.adapter_kind,
            model_id=initial_state.model_id,
            model_revision=initial_state.model_revision,
            training_scope=initial_state.training_scope,
            dimension_scales=initial_state.dimension_scales,
            updated_at=initial_state.updated_at,
        )
    )
    _SimulationEmbeddingAdapterFactory.adapter = adapter
    input_repository = (
        prototype_rebuild_input_repository.PrototypeRebuildInputRepository(
            state_root=output_dir / "main_server" / "prototype_rebuild_inputs"
        )
    )
    _store_prototype_rebuild_input(
        rows=dataset_split.bootstrap_rows,
        embedding_spec=embedding_spec,
        repository=input_repository,
        rebuild_config=effective_rebuild_config,
    )
    stored_rebuild_service = _build_prototype_rebuild_runtime_service(
        output_dir=output_dir,
        build_strategy=prototype_build_strategy,
        input_repository=input_repository,
    )
    lifecycle_service = RoundLifecycleService(
        round_repository=round_repository,
        round_manager_service=round_manager,
        prototype_rebuild_runtime_service=stored_rebuild_service,
    )
    active_prototype = _rebuild_reference_prototype_pack(
        stored_rebuild_service=stored_rebuild_service,
        adapter_state=initial_state,
        prototype_version=initial_prototype_version,
        embedding_model_id=model_id,
        embedding_model_revision=initial_model_revision,
        built_at=now,
    )
    save_prototype_pack(output_dir, active_prototype)
    active_manifest = ModelManifest(
        schema_version="model_manifest.v1",
        model_id=model_id,
        model_revision=initial_model_revision,
        published_at=now,
        artifact_kind="shared_adapter_state",
        artifact_ref=str(initial_state_path),
        prototype_version=initial_prototype_version,
        training_scope=training_scope,
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
        base_model_id=embedding_spec.model_id,
        base_model_revision=embedding_spec.revision,
        notes="round_active_pair_only bootstrap manifest",
    )
    save_model_manifest(output_dir, active_manifest)

    initial_validation = evaluate_rows(
        rows=validation_rows,
        adapter=adapter,
        adapter_state=initial_state,
        prototype_pack=active_prototype,
        model_id=model_id,
        scoring_service=validation_scoring_service,
        confidence_threshold=effective_validation_config.confidence_threshold,
        margin_threshold=effective_validation_config.margin_threshold,
    )

    round_summaries: list[SimulationRoundSummary] = []
    active_state = initial_state
    for round_index in range(1, rounds + 1):
        round_id = f"round_{round_index:04d}"
        round_record = lifecycle_service.open_round(
            _build_round_open_request(
                active_manifest=active_manifest,
                round_id=round_id,
                training_task_config=effective_training_task_config,
            )
        )
        training_task = round_record.training_task
        training_scoring_service = ScoringService.from_objective_config(
            training_task.objective_config,
            similarity_name=effective_validation_config.similarity_name,
        )

        updates = []
        client_summaries: list[ClientRoundSummary] = []
        for shard in dataset_split.client_shards:
            training_examples = build_training_examples(
                rows=shard.rows,
                adapter=adapter,
                adapter_state=active_state,
                prototype_pack=active_prototype,
                model_id=model_id,
                scoring_service=training_scoring_service,
            )
            local_training_service = LocalTrainingService(
                repository=TrainingArtifactRepository(
                    state_root=output_dir / "agents" / shard.client_id
                )
            )
            local_result = local_training_service.run(
                LocalTrainingRequest(
                    training_examples=training_examples,
                    training_task=training_task,
                    model_manifest=active_manifest,
                )
            )
            save_selection_diagnostics(
                output_dir=output_dir,
                round_id=round_id,
                client_id=shard.client_id,
                rows=shard.rows,
                training_examples=training_examples,
                selection_result=local_result.selection_result,
                diagnostics_config=effective_diagnostics_config,
            )
            if local_result.update_envelope is not None:
                lifecycle_service.accept_update(
                    round_id,
                    local_result.update_envelope,
                )
                updates.append(local_result.update_envelope)
            client_summaries.append(
                ClientRoundSummary(
                    client_id=shard.client_id,
                    candidate_count=local_result.selection_result.total_count,
                    accepted_count=local_result.selection_result.accepted_count,
                    update_generated=local_result.update_envelope is not None,
                )
            )

        if not updates:
            break

        next_model_revision = f"sim_rev_{round_index:04d}"
        next_prototype_version = f"proto_sim_{round_index:04d}"
        finalized_round = lifecycle_service.finalize_round(
            round_id,
            RoundFinalizeRequest(
                next_model_revision=next_model_revision,
                next_prototype_version=next_prototype_version,
            ),
        )
        if finalized_round.publication is None:
            raise ValueError("Finalized simulation round must contain publication.")
        active_manifest = finalized_round.publication.next_manifest
        active_state = _load_active_state(
            manifest=active_manifest,
            state_repository=state_repository,
            round_manager=round_manager,
        )
        save_model_manifest(output_dir, active_manifest)
        if finalized_round.publication.prototype_pack_ref is None:
            raise ValueError(
                "Simulation finalize must publish a prototype pack reference."
            )
        active_prototype = load_prototype_pack_payload(
            Path(finalized_round.publication.prototype_pack_ref)
        )
        save_prototype_pack(output_dir, active_prototype)
        validation = evaluate_rows(
            rows=validation_rows,
            adapter=adapter,
            adapter_state=active_state,
            prototype_pack=active_prototype,
            model_id=model_id,
            scoring_service=validation_scoring_service,
            confidence_threshold=effective_validation_config.confidence_threshold,
            margin_threshold=effective_validation_config.margin_threshold,
        )
        round_summaries.append(
            SimulationRoundSummary(
                round_id=round_id,
                model_revision=next_model_revision,
                prototype_version=next_prototype_version,
                update_count=len(updates),
                validation=validation,
                clients=tuple(client_summaries),
            )
        )

    final_validation = (
        round_summaries[-1].validation if round_summaries else initial_validation
    )
    return SimulationResult(
        initial_model_revision=initial_model_revision,
        initial_prototype_version=initial_prototype_version,
        initial_validation=initial_validation,
        final_validation=final_validation,
        rounds=tuple(round_summaries),
    )


def parse_created_at(value: str) -> datetime:
    """row의 created_at 문자열을 timezone-aware datetime으로 바꾼다."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _build_round_open_request(
    *,
    active_manifest: ModelManifest,
    round_id: str,
    training_task_config: FederatedTrainingTaskConfig,
) -> RoundOpenRequest:
    return RoundOpenRequest(
        active_manifest=active_manifest,
        round_id=round_id,
        local_epochs=int(training_task_config.local_epochs),
        batch_size=int(training_task_config.batch_size),
        learning_rate=float(training_task_config.learning_rate),
        max_steps=int(training_task_config.max_steps),
        objective_config=dict(training_task_config.objective_config),
        selection_policy=dict(training_task_config.selection_policy),
        min_required_examples=int(training_task_config.min_required_examples),
        gradient_clip_norm=(
            None
            if training_task_config.gradient_clip_norm is None
            else float(training_task_config.gradient_clip_norm)
        ),
    )


def _build_legacy_task_config(
    *,
    confidence_threshold: float | None,
    margin_threshold: float | None,
    max_examples: int | None,
    min_required_examples: int | None,
    gradient_clip_norm: float | None,
) -> FederatedTrainingTaskConfig:
    return FederatedTrainingTaskConfig(
        min_required_examples=min_required_examples or 1,
        gradient_clip_norm=gradient_clip_norm,
        objective_config={
            "loss": "diagonal_scale_heuristic",
            "confidence_threshold": _resolve_threshold(
                confidence_threshold,
                fallback=None,
                default=0.6,
            ),
            "margin_threshold": _resolve_threshold(
                margin_threshold,
                fallback=None,
                default=0.02,
            ),
            "score_policy_name": "max_cosine",
            "acceptance_policy_name": "top1_margin_threshold",
            "privacy_guard_name": "diagonal_scale_clip_only",
        },
        selection_policy={"max_examples": max_examples or 128},
    )


def _build_validation_scoring_service(
    validation_config: FederatedValidationConfig,
) -> ScoringService:
    return ScoringService.from_objective_config(
        TrainingObjectiveConfig.from_mapping(
            {
                "loss": "diagonal_scale_heuristic",
                "score_policy_name": validation_config.score_policy_name,
                **(
                    {}
                    if validation_config.score_top_k is None
                    else {"score_top_k": validation_config.score_top_k}
                ),
            }
        ),
        similarity_name=validation_config.similarity_name,
    )


def _resolve_threshold(
    value: float | None,
    *,
    fallback: object,
    default: float,
) -> float:
    if value is not None:
        return float(value)
    if fallback is None:
        return default
    if isinstance(fallback, bool):
        raise ValueError("Threshold config must not be bool.")
    return float(fallback)


def _resolve_optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("score_top_k must not be bool.")
    parsed = int(value)
    if parsed < 1:
        raise ValueError("score_top_k must be positive.")
    return parsed
