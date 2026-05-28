"""PrototypeRebuildService unit tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from main_server.src.infrastructure.repositories import (
    prototype_build_state_repository as prototype_build_state_repository_module,
)
from main_server.src.infrastructure.repositories import (
    prototype_pack_repository as prototype_pack_repository_module,
)
from main_server.src.infrastructure.repositories import (
    prototype_rebuild_input_repository as prototype_rebuild_input_repository_module,
)
from main_server.src.services.federation.prototypes import (
    models as prototype_models,
)
from main_server.src.services.federation.prototypes import (
    prototype_build_state_service as build_state_service_module,
)
from main_server.src.services.federation.prototypes import (
    prototype_pack_service as pack_service_module,
)
from main_server.src.services.federation.prototypes import (
    prototype_rebuild_service as rebuild_service_module,
)
from main_server.src.services.federation.prototypes import (
    publication_strategies as publication_strategy_module,
)
from main_server.src.services.federation.prototypes import (
    stored_input_rebuild_service as stored_rebuild_service_module,
)
from methods.prototype.building.base import (
    PrototypeBuildRequest,
)
from methods.prototype.building.kmeans import (
    KMeansPrototypeBuildStrategy,
)
from methods.prototype.building.single import (
    SinglePrototypeBuildStrategy,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    ClassifierHeadAdapterStatePayload,
)
from shared.src.domain.services.clock import FixedClock
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec

InMemoryPrototypePublicationStrategy = (
    publication_strategy_module.InMemoryPrototypePublicationStrategy
)
PrototypeBuildStateService = build_state_service_module.PrototypeBuildStateService
PrototypePackService = pack_service_module.PrototypePackService
PrototypeRebuildInputRecord = prototype_models.PrototypeRebuildInputRecord
PrototypeRebuildService = rebuild_service_module.PrototypeRebuildService
ReferencePrototypeRebuildRequest = prototype_models.ReferencePrototypeRebuildRequest
ServerReferencePrototypeSourceRow = prototype_models.ServerReferencePrototypeSourceRow
ReferenceRebuildPrototypePublicationStrategy = (
    publication_strategy_module.ReferenceRebuildPrototypePublicationStrategy
)
StoredReferencePrototypeRebuildRequest = (
    prototype_models.StoredReferencePrototypeRebuildRequest
)
StoredReferencePrototypeRebuildService = (
    stored_rebuild_service_module.StoredReferencePrototypeRebuildService
)


class _StaticEmbeddingAdapter:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vectors[text]) for text in texts]


class _StaticEmbeddingAdapterFactory:
    _vectors: dict[str, list[float]] = {}

    @classmethod
    def create(cls, spec: EmbeddingAdapterSpec) -> _StaticEmbeddingAdapter:
        del spec
        return _StaticEmbeddingAdapter(cls._vectors)


def _build_request(
    *,
    embeddings_by_category: dict[str, tuple[tuple[float, ...], ...]],
    prototype_version: str,
) -> PrototypeBuildRequest:
    return PrototypeBuildRequest(
        embeddings_by_category=embeddings_by_category,
        prototype_version=prototype_version,
        embedding_backend="hash_debug",
        embedding_model_id="hash_debug",
        embedding_model_revision="test",
        mapping_version="ourafla_to_4cat.v1",
        built_at=None,
    )


def _build_service(
    *,
    tmp_path: Path,
    build_strategy: SinglePrototypeBuildStrategy | KMeansPrototypeBuildStrategy,
    fixed_time: datetime,
) -> tuple[
    PrototypeRebuildService,
    prototype_pack_repository_module.PrototypePackRepository,
    prototype_build_state_repository_module.PrototypeBuildStateRepository,
]:
    pack_repository = prototype_pack_repository_module.PrototypePackRepository(
        state_root=tmp_path / "main_server" / "prototype_packs"
    )
    build_state_repository = (
        prototype_build_state_repository_module.PrototypeBuildStateRepository(
            state_root=tmp_path / "main_server" / "prototype_build_states"
        )
    )
    publication_strategy = ReferenceRebuildPrototypePublicationStrategy(
        reference_pack_output_dir=tmp_path / "reference" / "prototype_packs",
        reference_build_state_output_dir=tmp_path
        / "reference"
        / "prototype_build_states",
        prototype_pack_service=PrototypePackService(repository=pack_repository),
        prototype_build_state_service=PrototypeBuildStateService(
            repository=build_state_repository
        ),
    )
    service = PrototypeRebuildService(
        build_strategy=build_strategy,
        publication_strategy=publication_strategy,
        clock=FixedClock(fixed_time),
    )
    return service, pack_repository, build_state_repository


def test_rebuild_service_publishes_single_pack_and_build_state(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, tzinfo=timezone.utc)
    service, pack_repository, build_state_repository = _build_service(
        tmp_path=tmp_path,
        build_strategy=SinglePrototypeBuildStrategy(),
        fixed_time=fixed_time,
    )

    result = service.rebuild(
        _build_request(
            embeddings_by_category={
                "anxiety": ((1.0, 3.0), (3.0, 5.0)),
                "normal": ((2.0, 2.0), (4.0, 4.0)),
            },
            prototype_version="proto_single_v1",
        )
    )

    assert result.pack_payload.built_at == fixed_time
    assert result.build_state_payload is not None
    assert (
        result.reference_pack_path is not None and result.reference_pack_path.exists()
    )
    assert (
        result.reference_build_state_path is not None
        and result.reference_build_state_path.exists()
    )
    assert result.published_pack_path.exists()
    assert result.published_build_state_path is not None
    assert result.published_build_state_path.exists()
    assert len(result.pack_payload.categories["anxiety"]) == 1
    assert (
        pack_repository.load_pack("proto_single_v1").prototype_version
        == "proto_single_v1"
    )
    assert (
        build_state_repository.load_state("proto_single_v1").prototype_version
        == "proto_single_v1"
    )


def test_rebuild_service_publishes_multi_pack_without_build_state(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, tzinfo=timezone.utc)
    service, pack_repository, build_state_repository = _build_service(
        tmp_path=tmp_path,
        build_strategy=KMeansPrototypeBuildStrategy(
            candidate_ks=(2,),
            silhouette_sample_size=10,
            random_state=0,
        ),
        fixed_time=fixed_time,
    )

    result = service.rebuild(
        _build_request(
            embeddings_by_category={
                "anxiety": (
                    (1.0, 0.0),
                    (1.0, 0.1),
                    (-1.0, 0.0),
                    (-1.0, -0.1),
                ),
                "normal": ((0.0, 1.0), (0.0, 1.1)),
            },
            prototype_version="proto_kmeans_v1",
        )
    )

    assert result.pack_payload.built_at == fixed_time
    assert result.build_state_payload is None
    assert (
        result.reference_pack_path is not None and result.reference_pack_path.exists()
    )
    assert result.reference_build_state_path is None
    assert result.published_pack_path.exists()
    assert result.published_build_state_path is None
    assert len(result.pack_payload.categories["anxiety"]) == 2
    assert len(pack_repository.load_pack("proto_kmeans_v1").categories["anxiety"]) == 2
    assert build_state_repository.has_state("proto_kmeans_v1") is False


def test_rebuild_service_rebuilds_from_reference_rows_with_builder_strategy() -> None:
    service = PrototypeRebuildService(
        build_strategy=KMeansPrototypeBuildStrategy(
            candidate_ks=(2,),
            silhouette_sample_size=10,
            random_state=0,
        ),
        publication_strategy=InMemoryPrototypePublicationStrategy(),
        clock=FixedClock(datetime(2026, 4, 2, tzinfo=timezone.utc)),
    )
    adapter = _StaticEmbeddingAdapter(
        {
            "cluster_a_1": [1.0, 0.0],
            "cluster_a_2": [1.0, 0.1],
            "cluster_b_1": [-1.0, 0.0],
            "cluster_b_2": [-1.0, -0.1],
            "normal_1": [0.0, 1.0],
            "normal_2": [0.0, 1.1],
        }
    )
    adapter_state = ClassifierHeadAdapterStatePayload.zero_initialized(
        model_id="tracemind-embed-sim",
        model_revision="sim_rev_0000",
        training_scope="adapter_only",
        labels=("anxiety", "normal"),
        embedding_dim=2,
        updated_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
    )

    result = service.rebuild_from_reference_rows(
        ReferencePrototypeRebuildRequest(
            rows=(
                ServerReferencePrototypeSourceRow(
                    text="cluster_a_1",
                    category="anxiety",
                ),
                ServerReferencePrototypeSourceRow(
                    text="cluster_a_2",
                    category="anxiety",
                ),
                ServerReferencePrototypeSourceRow(
                    text="cluster_b_1",
                    category="anxiety",
                ),
                ServerReferencePrototypeSourceRow(
                    text="cluster_b_2",
                    category="anxiety",
                ),
                ServerReferencePrototypeSourceRow(
                    text="normal_1",
                    category="normal",
                ),
                ServerReferencePrototypeSourceRow(
                    text="normal_2",
                    category="normal",
                ),
            ),
            adapter=adapter,
            adapter_state=adapter_state,
            prototype_version="proto_sim_0000",
            embedding_model_id="tracemind-embed-sim",
            embedding_model_revision="sim_rev_0000",
            embedding_backend="simulation",
            mapping_version="ourafla_to_4cat.v1",
            built_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        )
    )

    assert result.published_pack_path is None
    assert result.build_state_payload is None
    assert len(result.pack_payload.categories["anxiety"]) == 2
    assert len(result.pack_payload.categories["normal"]) == 1


def test_rebuild_service_applies_reference_row_metadata_override() -> None:
    service = PrototypeRebuildService(
        build_strategy=SinglePrototypeBuildStrategy(),
        publication_strategy=InMemoryPrototypePublicationStrategy(),
        clock=FixedClock(datetime(2026, 4, 2, tzinfo=timezone.utc)),
    )
    adapter = _StaticEmbeddingAdapter(
        {
            "cluster_a_1": [1.0, 0.0],
            "cluster_a_2": [1.0, 0.1],
        }
    )
    adapter_state = ClassifierHeadAdapterStatePayload.zero_initialized(
        model_id="tracemind-embed-sim",
        model_revision="sim_rev_0000",
        training_scope="adapter_only",
        labels=("anxiety", "normal"),
        embedding_dim=2,
        updated_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
    )

    result = service.rebuild_from_reference_rows(
        ReferencePrototypeRebuildRequest(
            rows=(
                ServerReferencePrototypeSourceRow(
                    text="cluster_a_1",
                    category="anxiety",
                ),
                ServerReferencePrototypeSourceRow(
                    text="cluster_a_2",
                    category="anxiety",
                ),
            ),
            adapter=adapter,
            adapter_state=adapter_state,
            prototype_version="proto_sim_0000",
            embedding_model_id="tracemind-embed-sim",
            embedding_model_revision="sim_rev_0000",
            embedding_backend="custom_simulation",
            mapping_version="custom_mapping.v1",
            built_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
            translation_model_id="toy-translator",
            translation_model_revision="r1",
            translation_direction="en->ko",
        )
    )

    assert result.pack_payload.build_method == "mean_centroid_l2_normalized"
    assert result.pack_payload.mapping_version == "custom_mapping.v1"
    assert result.pack_payload.translation_model_id == "toy-translator"
    assert result.pack_payload.translation_model_revision == "r1"
    assert result.pack_payload.translation_direction == "en->ko"


def test_rebuild_input_repository_saves_and_loads_active_input(tmp_path: Path) -> None:
    repository = (
        prototype_rebuild_input_repository_module.PrototypeRebuildInputRepository(
            state_root=tmp_path / "prototype_rebuild_inputs"
        )
    )
    record = PrototypeRebuildInputRecord(
        input_id="bootstrap_v1",
        embedding_spec=EmbeddingAdapterSpec(
            backend="hash_debug",
            model_id="hash_debug",
            revision="seed",
            hash_dim=8,
        ),
        rows=(
            ServerReferencePrototypeSourceRow(text="alpha", category="anxiety"),
            ServerReferencePrototypeSourceRow(text="beta", category="normal"),
        ),
        mapping_version="ourafla_to_4cat.v1",
        required_categories=("anxiety", "normal"),
    )

    repository.save_input(record)
    repository.set_active(
        "bootstrap_v1",
        activated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )
    loaded = repository.load_active_input()

    assert loaded.input_id == "bootstrap_v1"
    assert loaded.embedding_spec.backend == "hash_debug"
    assert tuple(row.text for row in loaded.rows) == ("alpha", "beta")
    assert loaded.required_categories == ("anxiety", "normal")


def test_server_reference_row_rejects_agent_query_source_kind() -> None:
    with pytest.raises(ValueError, match="Agent query/raw text"):
        ServerReferencePrototypeSourceRow(
            text="agent raw query",
            category="anxiety",
            source_kind="query_buffer",  # type: ignore[arg-type]
        )


def test_rebuild_input_payload_rejects_query_buffer_source_kind() -> None:
    row_payload = (
        prototype_rebuild_input_repository_module.PrototypeRebuildInputRowPayload
    )

    with pytest.raises(ValidationError):
        row_payload.model_validate(
            {
                "text": "agent raw query",
                "category": "anxiety",
                "source_kind": "query_buffer",
            }
        )


def test_stored_reference_rebuild_service_rebuilds_from_active_input(
    tmp_path: Path,
) -> None:
    repository = (
        prototype_rebuild_input_repository_module.PrototypeRebuildInputRepository(
            state_root=tmp_path / "prototype_rebuild_inputs"
        )
    )
    repository.save_input(
        PrototypeRebuildInputRecord(
            input_id="bootstrap_v1",
            embedding_spec=EmbeddingAdapterSpec(
                backend="hash_debug",
                model_id="hash_debug",
                revision="seed",
                hash_dim=8,
            ),
            rows=(
                ServerReferencePrototypeSourceRow(
                    text="cluster_a_1",
                    category="anxiety",
                ),
                ServerReferencePrototypeSourceRow(
                    text="cluster_a_2",
                    category="anxiety",
                ),
                ServerReferencePrototypeSourceRow(text="normal_1", category="normal"),
            ),
            mapping_version="ourafla_to_4cat.v1",
        )
    )
    repository.set_active(
        "bootstrap_v1",
        activated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )
    _StaticEmbeddingAdapterFactory._vectors = {
        "cluster_a_1": [1.0, 0.0],
        "cluster_a_2": [1.0, 0.1],
        "normal_1": [0.0, 1.0],
    }
    stored_service = StoredReferencePrototypeRebuildService(
        input_repository=repository,
        prototype_rebuild_service=PrototypeRebuildService(
            build_strategy=SinglePrototypeBuildStrategy(),
            publication_strategy=InMemoryPrototypePublicationStrategy(),
            clock=FixedClock(datetime(2026, 4, 2, tzinfo=timezone.utc)),
        ),
        adapter_factory=_StaticEmbeddingAdapterFactory,
    )
    adapter_state = ClassifierHeadAdapterStatePayload.zero_initialized(
        model_id="tracemind-embed-sim",
        model_revision="sim_rev_0001",
        training_scope="adapter_only",
        labels=("anxiety", "normal"),
        embedding_dim=2,
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )

    result = stored_service.rebuild(
        StoredReferencePrototypeRebuildRequest(
            adapter_state=adapter_state,
            prototype_version="proto_sim_0001",
            embedding_model_id="tracemind-embed-sim",
            embedding_model_revision="sim_rev_0001",
            built_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
        )
    )

    assert result.source_input_id == "bootstrap_v1"
    assert result.pack_payload.prototype_version == "proto_sim_0001"
    assert result.pack_payload.mapping_version == "ourafla_to_4cat.v1"
    assert len(result.pack_payload.categories["anxiety"]) == 1
