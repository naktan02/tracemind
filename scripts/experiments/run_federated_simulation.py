"""합성 federation simulation을 실행한다."""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_SERVER_ROOT = PROJECT_ROOT / "main-server"
if str(MAIN_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_SERVER_ROOT))

from src.infrastructure.repositories.vector_adapter_state_repository import (  # noqa: E402
    SharedAdapterStateRepository,
)
from src.services.rounds.round_manager_service import (  # noqa: E402
    RoundManagerService,
    RoundPublicationRequest,
    TrainingTaskRequest,
)

from agent.src.infrastructure.model_adapters.embedding.factory import (  # noqa: E402
    EmbeddingAdapterFactory,
    EmbeddingAdapterSpec,
)
from agent.src.infrastructure.repositories.training_artifact_repository import (  # noqa: E402
    TrainingArtifactRepository,
)
from agent.src.services.inference.scoring_service import ScoringService  # noqa: E402
from agent.src.services.training.local_training_service import (  # noqa: E402
    EmbeddedTrainingExample,
    LocalTrainingRequest,
    LocalTrainingService,
)
from scripts.prototypes.prototype_pack_builder import PrototypePackBuilder  # noqa: E402
from shared.src.contracts.adapter_contracts import (  # noqa: E402
    DiagonalScaleAdapterStatePayload,
)
from shared.src.contracts.model_contracts import (  # noqa: E402
    ModelManifestPayload,
    dump_model_manifest_payload,
)
from shared.src.contracts.prototype_contracts import (  # noqa: E402
    PrototypePackPayload,
    dump_prototype_pack_payload,
    extract_category_centroids,
)
from shared.src.domain.entities.artifacts.model_manifest import (  # noqa: E402
    ModelManifest,
)
from shared.src.domain.entities.inference.events import ScoredEvent  # noqa: E402
from shared.src.domain.entities.training.vector_adapter_state import (  # noqa: E402
    VectorAdapterState,
)


@dataclass(slots=True)
class FederatedClientShard:
    """한 client에 할당된 train row 묶음."""

    client_id: str
    rows: list[dict[str, Any]]


@dataclass(slots=True)
class FederatedDatasetSplit:
    """bootstrap과 client shard로 나눈 train subset."""

    bootstrap_rows: list[dict[str, Any]]
    client_shards: tuple[FederatedClientShard, ...]


@dataclass(slots=True)
class ClientRoundSummary:
    """client 하나의 라운드 참여 요약."""

    client_id: str
    candidate_count: int
    accepted_count: int
    update_generated: bool


@dataclass(slots=True)
class SimulationEvaluation:
    """validation 평가 결과."""

    row_count: int
    top1_accuracy: float
    accepted_ratio: float


@dataclass(slots=True)
class SimulationRoundSummary:
    """한 라운드 종료 후 요약."""

    round_id: str
    model_revision: str
    prototype_version: str
    update_count: int
    validation: SimulationEvaluation
    clients: tuple[ClientRoundSummary, ...]


@dataclass(slots=True)
class SimulationResult:
    """전체 simulation 요약."""

    initial_model_revision: str
    initial_prototype_version: str
    initial_validation: SimulationEvaluation
    final_validation: SimulationEvaluation
    rounds: tuple[SimulationRoundSummary, ...]


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """JSONL 파일을 row 목록으로 읽는다."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def split_rows_for_federation(
    rows: list[dict[str, Any]],
    *,
    bootstrap_ratio: float,
    client_count: int,
    seed: int,
) -> FederatedDatasetSplit:
    """train row를 prototype bootstrap과 non-IID client shard로 나눈다."""
    if not 0.0 < bootstrap_ratio < 1.0:
        raise ValueError("bootstrap_ratio must be between 0 and 1.")
    if client_count <= 0:
        raise ValueError("client_count must be positive.")

    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_label[row["mapped_label_4"]].append(row)

    rng = random.Random(seed)
    bootstrap_rows: list[dict[str, Any]] = []
    remaining_by_label: dict[str, list[dict[str, Any]]] = {}
    for label in sorted(rows_by_label):
        bucket = list(rows_by_label[label])
        rng.shuffle(bucket)
        bootstrap_count = int(round(len(bucket) * bootstrap_ratio))
        if bootstrap_count <= 0 and len(bucket) > 1:
            bootstrap_count = 1
        if bootstrap_count >= len(bucket):
            bootstrap_count = len(bucket) - 1
        bootstrap_rows.extend(bucket[:bootstrap_count])
        remaining_by_label[label] = bucket[bootstrap_count:]

    client_shards = [
        FederatedClientShard(client_id=f"agent_{index + 1:02d}", rows=[])
        for index in range(client_count)
    ]
    labels = sorted(remaining_by_label)
    for label_index, label in enumerate(labels):
        bucket = list(remaining_by_label[label])
        dominant_index = label_index % client_count
        secondary_index = (
            (label_index + 1) % client_count if client_count > 1 else dominant_index
        )
        split_point = int(len(bucket) * 0.75)
        if split_point <= 0:
            split_point = len(bucket)
        client_shards[dominant_index].rows.extend(bucket[:split_point])
        if secondary_index != dominant_index:
            client_shards[secondary_index].rows.extend(bucket[split_point:])

    for shard in client_shards:
        rng.shuffle(shard.rows)

    return FederatedDatasetSplit(
        bootstrap_rows=bootstrap_rows,
        client_shards=tuple(client_shards),
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
    confidence_threshold: float,
    margin_threshold: float,
    max_examples: int,
    min_required_examples: int,
    gradient_clip_norm: float,
) -> SimulationResult:
    """bootstrap -> client pseudo-label -> aggregate -> republish 루프를 실행한다."""
    dataset_split = split_rows_for_federation(
        train_rows,
        bootstrap_ratio=bootstrap_ratio,
        client_count=client_count,
        seed=seed,
    )
    adapter = EmbeddingAdapterFactory.create(embedding_spec)
    if not dataset_split.bootstrap_rows:
        raise ValueError("Bootstrap split must contain at least one row.")
    embedding_dim = len(
        adapter.embed_texts([str(dataset_split.bootstrap_rows[0]["text"])])[0]
    )
    state_repository = SharedAdapterStateRepository(
        state_root=output_dir / "main_server" / "shared_adapter_states"
    )
    round_manager = RoundManagerService(artifact_repository=state_repository)

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
    active_prototype = build_prototype_pack_from_rows(
        rows=dataset_split.bootstrap_rows,
        adapter=adapter,
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
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )

    round_summaries: list[SimulationRoundSummary] = []
    active_state = initial_state
    for round_index in range(1, rounds + 1):
        round_id = f"round_{round_index:04d}"
        training_task = round_manager.create_training_task(
            TrainingTaskRequest(
                active_manifest=active_manifest,
                round_id=round_id,
                objective_config={
                    "loss": "diagonal_scale_heuristic",
                    "confidence_threshold": confidence_threshold,
                    "margin_threshold": margin_threshold,
                },
                selection_policy={"max_examples": max_examples},
                min_required_examples=min_required_examples,
                gradient_clip_norm=gradient_clip_norm,
            )
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
            if local_result.update_envelope is not None:
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
        publication = round_manager.publish_next_pair(
            RoundPublicationRequest(
                base_manifest=active_manifest,
                updates=updates,
                next_model_revision=next_model_revision,
                next_prototype_version=next_prototype_version,
            )
        )
        active_state = publication.next_state
        active_manifest = publication.next_manifest
        save_model_manifest(output_dir, active_manifest)
        active_prototype = build_prototype_pack_from_rows(
            rows=dataset_split.bootstrap_rows,
            adapter=adapter,
            adapter_state=active_state,
            prototype_version=next_prototype_version,
            embedding_model_id=model_id,
            embedding_model_revision=next_model_revision,
            built_at=active_manifest.published_at,
        )
        save_prototype_pack(output_dir, active_prototype)
        validation = evaluate_rows(
            rows=validation_rows,
            adapter=adapter,
            adapter_state=active_state,
            prototype_pack=active_prototype,
            model_id=model_id,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
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


def build_training_examples(
    *,
    rows: list[dict[str, Any]],
    adapter: Any,
    adapter_state: VectorAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
) -> tuple[EmbeddedTrainingExample, ...]:
    """row를 scored event + embedding 예시로 변환한다."""
    if not rows:
        return ()

    texts = [str(row["text"]) for row in rows]
    base_embeddings = adapter.embed_texts(texts)
    centroids = extract_category_centroids(prototype_pack)
    scoring_service = ScoringService()
    examples: list[EmbeddedTrainingExample] = []
    for row, base_embedding in zip(rows, base_embeddings, strict=True):
        adapted_embedding = adapter_state.apply(base_embedding)
        scored_event = ScoredEvent(
            query_id=str(row["query_id"]),
            occurred_at=parse_created_at(row["created_at"]),
            translated_text=None,
            embedding_model_id=model_id,
            translation_model_id=prototype_pack.translation_model_id,
            category_scores=scoring_service.score(adapted_embedding, centroids),
        )
        examples.append(
            EmbeddedTrainingExample(
                scored_event=scored_event,
                embedding=adapted_embedding,
                base_embedding=list(base_embedding),
            )
        )
    return tuple(examples)


def evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    adapter: Any,
    adapter_state: VectorAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
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
        if ranked_scores[0][0] == row["mapped_label_4"]:
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


def build_prototype_pack_from_rows(
    *,
    rows: list[dict[str, Any]],
    adapter: Any,
    adapter_state: VectorAdapterState,
    prototype_version: str,
    embedding_model_id: str,
    embedding_model_revision: str,
    built_at: datetime,
) -> PrototypePackPayload:
    """bootstrap row에서 현재 model revision용 PrototypePack을 다시 만든다."""
    if not rows:
        raise ValueError("rows must not be empty when building a PrototypePack.")

    texts = [str(row["text"]) for row in rows]
    base_embeddings = adapter.embed_texts(texts)
    embeddings_by_category: dict[str, list[list[float]]] = defaultdict(list)
    for row, base_embedding in zip(rows, base_embeddings, strict=True):
        embeddings_by_category[row["mapped_label_4"]].append(
            adapter_state.apply(base_embedding)
        )

    builder = PrototypePackBuilder()
    pack = builder.build(
        embeddings_by_category,
        prototype_version=prototype_version,
        embedding_backend="simulation",
        embedding_model_id=embedding_model_id,
        embedding_model_revision=embedding_model_revision,
        translation_model_id=None,
        translation_model_revision=None,
        translation_direction=None,
        mapping_version="ourafla_to_4cat.v1",
        built_at=built_at,
        required_categories=tuple(sorted(embeddings_by_category)),
    )
    return PrototypePackPayload.model_validate(
        {
            "schema_version": pack.schema_version,
            "prototype_version": pack.prototype_version,
            "embedding_model_id": pack.embedding_model_id,
            "embedding_model_revision": pack.embedding_model_revision,
            "translation_model_id": pack.translation_model_id,
            "translation_model_revision": pack.translation_model_revision,
            "translation_direction": pack.translation_direction,
            "mapping_version": pack.mapping_version,
            "build_method": pack.build_method,
            "distance_metric": pack.distance_metric,
            "built_at": pack.built_at,
            "categories": {
                category: {
                    "centroid": prototype.centroid,
                    "sample_count": prototype.sample_count,
                }
                for category, prototype in pack.categories.items()
            },
        }
    )


def save_prototype_pack(output_dir: Path, payload: PrototypePackPayload) -> Path:
    """prototype pack payload를 output_dir 아래에 저장한다."""
    path = (
        output_dir
        / "main_server"
        / "prototype_packs"
        / f"{payload.prototype_version}.json"
    )
    dump_prototype_pack_payload(path, payload)
    return path


def save_model_manifest(output_dir: Path, manifest: ModelManifest) -> Path:
    """model manifest entity를 output_dir 아래 JSON으로 저장한다."""
    path = (
        output_dir
        / "main_server"
        / "model_manifests"
        / f"{manifest.model_revision}.json"
    )
    dump_model_manifest_payload(
        path,
        ModelManifestPayload(
            schema_version=manifest.schema_version,
            model_id=manifest.model_id,
            model_revision=manifest.model_revision,
            published_at=manifest.published_at,
            artifact_kind=manifest.artifact_kind,
            artifact_ref=manifest.artifact_ref,
            prototype_version=manifest.prototype_version,
            training_scope=manifest.training_scope,
            training_enabled=manifest.training_enabled,
            compatible_task_types=list(manifest.compatible_task_types),
            base_model_id=manifest.base_model_id,
            base_model_revision=manifest.base_model_revision,
            translation_model_id=manifest.translation_model_id,
            translation_model_revision=manifest.translation_model_revision,
            notes=manifest.notes,
        ),
    )
    return path


def parse_created_at(value: str) -> datetime:
    """row의 created_at 문자열을 timezone-aware datetime으로 바꾼다."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiments/run_federated_simulation",
)
def main(cfg: DictConfig) -> None:
    embedding_spec = instantiate(cfg.embedding.spec)
    result = run_simulation(
        train_rows=load_jsonl_rows(Path(str(cfg.train_jsonl))),
        validation_rows=load_jsonl_rows(Path(str(cfg.validation_jsonl))),
        output_dir=Path(str(cfg.federated_run_preset.output_dir)),
        client_count=int(cfg.federated_run_preset.client_count),
        rounds=int(cfg.federated_run_preset.rounds),
        bootstrap_ratio=float(cfg.federated_run_preset.bootstrap_ratio),
        seed=int(cfg.seed),
        embedding_spec=embedding_spec,
        model_id=str(cfg.published_model_id),
        training_scope=str(cfg.training_scope),
        confidence_threshold=float(cfg.confidence_threshold),
        margin_threshold=float(cfg.margin_threshold),
        max_examples=int(cfg.federated_run_preset.max_examples),
        min_required_examples=int(cfg.federated_run_preset.min_required_examples),
        gradient_clip_norm=float(cfg.gradient_clip_norm),
    )

    print(f"initial_model_revision={result.initial_model_revision}")
    print(f"initial_prototype_version={result.initial_prototype_version}")
    print(
        "initial_validation="
        f"accuracy:{result.initial_validation.top1_accuracy:.4f},"
        f"accepted_ratio:{result.initial_validation.accepted_ratio:.4f}"
    )
    if result.rounds:
        last_round = result.rounds[-1]
        print(f"final_model_revision={last_round.model_revision}")
        print(f"final_prototype_version={last_round.prototype_version}")
        print(
            "final_validation="
            f"accuracy:{result.final_validation.top1_accuracy:.4f},"
            f"accepted_ratio:{result.final_validation.accepted_ratio:.4f}"
        )
        print(f"round_count={len(result.rounds)}")
    else:
        print("round_count=0")
        print("note=no client updates satisfied the pseudo-label selection criteria.")


if __name__ == "__main__":
    main()
