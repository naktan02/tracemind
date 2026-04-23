"""ScoringService unit tests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent.src.services.inference.scoring_backends import (
    PROTOTYPE_SIMILARITY_BACKEND_NAME,
    PROTOTYPE_SIMILARITY_CONFIDENCE_KIND,
    PrototypeSimilarityScoringBackend,
    register_scoring_backend,
)
from agent.src.services.inference.scoring_policies import TopKMeanCosineScorePolicy
from agent.src.services.inference.scoring_service import ScoringService
from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


def _load_prototypes() -> dict[str, list[float]]:
    fixture_path = Path(__file__).with_name("fixtures") / "prototype_pack_v1.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return {
        category: values["centroid"]
        for category, values in payload["categories"].items()
    }


def test_score_returns_cosine_similarity_for_each_category() -> None:
    service = ScoringService()

    scores = service.score([1.0, 0.0, 0.0], _load_prototypes())

    assert scores["anxiety"] == pytest.approx(1.0)
    assert scores["depression"] == pytest.approx(0.0)
    assert scores["suicidal"] == pytest.approx(-1.0)
    assert scores["normal"] == pytest.approx(1.0 / math.sqrt(2.0))


def test_score_uses_best_matching_prototype_for_multi_prototype_category() -> None:
    service = ScoringService()

    scores = service.score(
        [1.0, 0.0],
        {
            "alert": ([0.0, 1.0], [1.0, 0.0]),
            "safe": ([0.0, -1.0],),
        },
    )

    assert scores["alert"] == pytest.approx(1.0)
    assert scores["safe"] == pytest.approx(0.0)


def test_score_rejects_unsupported_similarity_metric() -> None:
    service = ScoringService(
        backend=PrototypeSimilarityScoringBackend(similarity_name="sigmoid")
    )

    with pytest.raises(ValueError, match="Unsupported similarity metric"):
        service.score([1.0, 0.0], {"anxiety": [1.0, 0.0]})


def test_score_rejects_dimension_mismatch() -> None:
    service = ScoringService()

    with pytest.raises(ValueError, match="dimensions must match"):
        service.score([1.0, 0.0, 0.0], {"anxiety": [1.0, 0.0]})


def test_score_rejects_zero_norm_embedding() -> None:
    service = ScoringService()

    with pytest.raises(ValueError, match="Embedding vector norm must be non-zero"):
        service.score([0.0, 0.0], {"anxiety": [1.0, 0.0]})


def test_score_rejects_zero_norm_prototype() -> None:
    service = ScoringService()

    with pytest.raises(ValueError, match="Prototype vector norm must be non-zero"):
        service.score([1.0, 0.0], {"anxiety": [0.0, 0.0]})


def test_score_can_switch_to_top_k_mean_policy() -> None:
    service = ScoringService(
        backend=PrototypeSimilarityScoringBackend(
            policy=TopKMeanCosineScorePolicy(top_k=2)
        )
    )

    scores = service.score(
        [1.0, 0.0],
        {
            "alert": ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]),
        },
    )

    assert scores["alert"] == pytest.approx(0.5)


def test_score_service_can_be_built_from_objective_config() -> None:
    service = ScoringService.from_objective_config(
        TrainingObjectiveConfig(
            scorer_backend_name="prototype_similarity",
            score_policy_name="topk_mean_cosine",
            score_top_k=2,
        )
    )

    scores = service.score(
        [1.0, 0.0],
        {
            "alert": ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]),
        },
    )

    assert scores["alert"] == pytest.approx(0.5)


def test_prototype_similarity_backend_keeps_fixed_implementation_name() -> None:
    backend = PrototypeSimilarityScoringBackend()

    assert backend.backend_name == PROTOTYPE_SIMILARITY_BACKEND_NAME
    assert backend.confidence_kind == PROTOTYPE_SIMILARITY_CONFIDENCE_KIND


def test_score_service_can_switch_registered_scoring_backend() -> None:
    @dataclass(slots=True)
    class _ConstantScoringBackend:
        backend_name: str = "constant_test_backend"
        confidence_kind: str = "constant_test_backend_top1"

        def score(self, embedding, prototypes, shared_state=None):
            del embedding, shared_state
            return {
                category: float(index + 1)
                for index, category in enumerate(sorted(prototypes))
            }

    register_scoring_backend(
        "constant_test_backend",
        factory=lambda _objective_config, _similarity_name: _ConstantScoringBackend(),
        catalog_entry=RegistryCatalogEntry(
            item_name="constant_test_backend",
            display_name="constant_test_backend",
            implementation_module=__name__,
            core_method_name="constant_test_backend",
            family_name="scoring",
            supported_adapter_kinds=("*",),
            metadata={"confidence_kind": "constant_test_backend_top1"},
        ),
    )
    service = ScoringService.from_objective_config(
        TrainingObjectiveConfig(
            scorer_backend_name="constant_test_backend",
        )
    )

    scores = service.score(
        [1.0, 0.0],
        {
            "alert": ([1.0, 0.0],),
            "safe": ([0.0, 1.0],),
        },
    )

    assert scores == {"alert": 1.0, "safe": 2.0}
    assert service.confidence_kind == "constant_test_backend_top1"
