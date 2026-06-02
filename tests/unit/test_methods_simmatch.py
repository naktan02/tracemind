"""Reusable SimMatch method core tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.algorithms.simmatch.memory_bank import SimMatchMemoryBank
from methods.ssl.algorithms.simmatch.simmatch import (
    SimMatchAlgorithm,
    compute_simmatch_step,
)
from methods.ssl.base import QuerySslStepContext
from methods.ssl.hooks.distribution_alignment import QueueDistributionAlignmentHook
from methods.ssl.registry import (
    build_query_ssl_algorithm,
    resolve_query_ssl_algorithm_descriptor,
)


class _FeatureClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(2))

    def extract_pooled_features(self, *, input_ids, attention_mask):
        del attention_mask
        values = input_ids.float()
        return torch.stack([values[:, 0], values[:, 1]], dim=1)

    def forward(self, *, input_ids, attention_mask):
        return self.classifier(
            self.extract_pooled_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )


class _IdentityProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(2))

    def forward(self, features):
        return self.projection(features)


def _labeled_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
        "row_indices": torch.tensor([0, 1], dtype=torch.long),
    }


def _unlabeled_batch() -> dict[str, torch.Tensor]:
    return {
        "weak_input_ids": torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
        "weak_attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "strong_input_ids": torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
        "strong_attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
    }


def _step_context(*, epoch_index: int) -> QuerySslStepContext:
    return QuerySslStepContext(
        epoch_index=epoch_index,
        step_index=1,
        global_step=epoch_index,
        total_train_steps=4,
        num_classes=2,
        device=torch.device("cpu"),
    )


def test_query_ssl_algorithm_registry_builds_simmatch_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="simmatch",
        parameters={
            "T": 0.5,
            "p_cutoff": 0.95,
            "proj_size": 2,
            "smoothing_alpha": 0.9,
            "da_len": 2,
            "in_loss_ratio": 1.0,
        },
    )

    assert isinstance(algorithm, SimMatchAlgorithm)
    assert algorithm.algorithm_name == "simmatch"
    assert algorithm.proj_size == 2


def test_query_ssl_algorithm_descriptor_exposes_simmatch_capability_spec() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("simmatch")

    assert descriptor.algorithm_name == "simmatch"
    assert descriptor.display_name == "SimMatch"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert descriptor.required_views.view_builder_name == "usb_multiview"
    assert descriptor.runtime_requirements.step_context_required is True


def test_compute_simmatch_step_updates_labeled_memory_bank_and_uses_in_loss() -> None:
    model = _FeatureClassifier()
    projection = _IdentityProjection()
    memory_bank = SimMatchMemoryBank(
        bank_size=2,
        feature_dim=2,
        device="cpu",
    )
    memory_bank.feature_bank = torch.eye(2)
    memory_bank.labels_bank = torch.tensor([0, 1], dtype=torch.long)
    dist_align = QueueDistributionAlignmentHook(
        num_classes=2,
        queue_length=2,
        p_target_type="uniform",
    )

    output = compute_simmatch_step(
        model=model,
        projection_head=projection,
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        dist_align_hook=dist_align,
        memory_bank=memory_bank,
        temperature=0.5,
        p_cutoff=0.0,
        smoothing_alpha=0.9,
        ema_bank=0.0,
        apply_similarity_smoothing=True,
        lambda_u=1.0,
        lambda_in=1.0,
    )

    assert output.loss_components["sup_loss"] > 0
    assert output.loss_components["unsup_loss"] >= 0
    assert output.loss_components["in_loss"] > 0
    assert output.metrics["util_ratio"] == 1.0
    assert torch.allclose(memory_bank.feature_bank, torch.eye(2))
    assert memory_bank.labels_bank.tolist() == [0, 1]


def test_simmatch_first_epoch_disables_similarity_loss_like_usb() -> None:
    model = _FeatureClassifier()
    algorithm = SimMatchAlgorithm(
        T=0.5,
        p_cutoff=0.0,
        proj_size=2,
        smoothing_alpha=0.9,
        da_len=2,
    )
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=2)
    algorithm.configure_labeled_dataset(labeled_row_count=2)
    algorithm.projection_head = _IdentityProjection()

    output = algorithm.compute_step_with_context(
        model=model,
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        step_context=_step_context(epoch_index=1),
    )

    assert torch.isclose(output.loss_components["in_loss"], torch.tensor(0.0))
    assert output.metrics["similarity_smoothing_applied"] == 0.0


def test_simmatch_algorithm_state_roundtrips_memory_bank() -> None:
    algorithm = SimMatchAlgorithm(
        T=0.5,
        p_cutoff=0.0,
        proj_size=2,
        smoothing_alpha=0.9,
        da_len=2,
    )
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=2)
    algorithm.configure_labeled_dataset(labeled_row_count=2)
    algorithm.projection_head = _IdentityProjection()
    algorithm.compute_step_with_context(
        model=_FeatureClassifier(),
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        step_context=_step_context(epoch_index=2),
    )
    state = algorithm.export_state()

    restored = SimMatchAlgorithm(
        T=0.5,
        p_cutoff=0.0,
        proj_size=2,
        smoothing_alpha=0.9,
        da_len=2,
    )
    restored.configure_dataset(num_classes=2, unlabeled_row_count=2)
    restored.configure_labeled_dataset(labeled_row_count=2)
    restored.load_state(state)
    restored.projection_head = _IdentityProjection()
    restored.compute_step_with_context(
        model=_FeatureClassifier(),
        labeled_batch=_labeled_batch(),
        unlabeled_batch=_unlabeled_batch(),
        step_context=_step_context(epoch_index=2),
    )
    restored_state = restored.export_state()

    assert torch.allclose(
        restored_state["memory_bank"]["feature_bank"],
        state["memory_bank"]["feature_bank"],
    )
    assert torch.equal(
        restored_state["memory_bank"]["labels_bank"],
        state["memory_bank"]["labels_bank"],
    )
