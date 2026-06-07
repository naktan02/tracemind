"""ScoringService unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from agent.src.services.inference.scoring_backends.registry import (
    list_scoring_backend_catalog_entries,
    register_scoring_backend,
)
from agent.src.services.inference.scoring_service import ScoringService
from shared.src.contracts.adapter_contract_families.classifier_head import (
    ClassifierHeadState,
)
from shared.src.contracts.common_types import TrainingScope
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


def test_score_service_can_switch_registered_scoring_backend() -> None:
    @dataclass(slots=True)
    class _ConstantScoringBackend:
        backend_name: str = "constant_test_backend"
        confidence_kind: str = "constant_test_backend_top1"

        def score(self, embedding, scoring_assets, shared_state=None):
            del embedding, shared_state
            return {
                category: float(index + 1)
                for index, category in enumerate(sorted(scoring_assets))
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
            training_backend_name="peft_classifier_trainer",
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


def test_score_service_uses_methods_owned_classifier_head_logits_backend() -> None:
    service = ScoringService.from_objective_config(
        TrainingObjectiveConfig(
            training_backend_name="peft_classifier_trainer",
            scorer_backend_name="classifier_head_logits",
        ),
        shared_state=ClassifierHeadState(
            schema_version="classifier_head_state.v1",
            adapter_kind="classifier_head",
            model_id="tracemind-embed",
            model_revision="rev_classifier_001",
            training_scope=TrainingScope.HEAD_ONLY,
            updated_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
            label_weights={
                "anxiety": [2.0, 0.0],
                "normal": [0.0, 1.0],
            },
            label_biases={"anxiety": 0.1, "normal": -0.1},
        ),
    )

    scores = service.score([0.5, 1.0], {})

    assert service.backend_name == "classifier_head_logits"
    assert service.confidence_kind == "classifier_head_logit_top1"
    assert scores["anxiety"] == pytest.approx(1.1)
    assert scores["normal"] == pytest.approx(0.9)


def test_score_service_default_backend_is_classifier_head_logits() -> None:
    service = ScoringService()

    assert service.backend_name == "classifier_head_logits"
    assert service.confidence_kind == "classifier_head_logit_top1"


def test_classifier_head_logits_catalog_points_to_classification_core() -> None:
    entries = {
        entry.item_name: entry for entry in list_scoring_backend_catalog_entries()
    }

    assert (
        entries["classifier_head_logits"].implementation_module
        == "methods.classification.linear_head.scoring"
    )
