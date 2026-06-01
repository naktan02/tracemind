"""Reusable CoMatch method core tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.algorithms.comatch.comatch import (
    CoMatchAlgorithm,
    comatch_contrastive_loss,
    compute_comatch_graph_targets,
    compute_comatch_step,
)
from methods.ssl.algorithms.comatch.memory_bank import CoMatchMemoryBank
from methods.ssl.hooks.distribution_alignment import QueueDistributionAlignmentHook
from methods.ssl.registry import (
    build_query_ssl_algorithm,
    resolve_query_ssl_algorithm_descriptor,
)


class _FeatureMapModel(nn.Module):
    classifier: nn.Linear

    def __init__(
        self,
        *,
        logits_by_token: dict[int, torch.Tensor],
        features_by_token: dict[int, torch.Tensor],
    ) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)
        self._logits_by_token = logits_by_token
        self._features_by_token = features_by_token

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        return torch.stack(
            [self._logits_by_token[int(row[0].item())] for row in input_ids]
        )

    def extract_pooled_features(self, *, input_ids, attention_mask):
        del attention_mask
        return torch.stack(
            [self._features_by_token[int(row[0].item())] for row in input_ids]
        )


def test_query_ssl_algorithm_registry_builds_comatch_algorithm() -> None:
    algorithm = build_query_ssl_algorithm(
        algorithm_name="comatch",
        parameters={
            "temperature": 0.5,
            "p_cutoff": 0.95,
            "contrast_p_cutoff": 0.8,
            "queue_batch": 4,
            "smoothing_alpha": 0.9,
            "da_len": 2,
            "proj_size": 2,
            "lambda_u": 1.0,
            "lambda_c": 1.0,
            "supervised_loss_weight": 1.0,
        },
    )

    assert isinstance(algorithm, CoMatchAlgorithm)
    assert algorithm.algorithm_name == "comatch"
    assert algorithm.proj_size == 2


def test_query_ssl_algorithm_descriptor_exposes_comatch_capabilities() -> None:
    descriptor = resolve_query_ssl_algorithm_descriptor("comatch")
    requirements = descriptor.runtime_requirements

    assert descriptor.algorithm_name == "comatch"
    assert descriptor.required_views.view_builder_name == "usb_weak_strong_pair"
    assert descriptor.required_views.view_names == ("text", "aug_0", "aug_1")
    assert requirements.batch_surface == "weak_strong_pair"
    assert requirements.model_outputs == frozenset({"logits", "pooled_features"})
    assert requirements.algorithm_state_surface == frozenset(
        {"feature_queue", "probability_queue"}
    )
    assert "auxiliary_trainable_module" in requirements.optimizer_lifecycle
    assert requirements.step_context_required is True


def test_comatch_graph_targets_apply_threshold_and_self_loop() -> None:
    probabilities = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]],
        dtype=torch.float32,
    )

    graph_targets = compute_comatch_graph_targets(
        probabilities=probabilities,
        contrast_p_cutoff=0.7,
    )

    raw = probabilities @ probabilities.t()
    raw.fill_diagonal_(1.0)
    raw = raw * raw.ge(0.7).to(raw.dtype)
    expected = raw / raw.sum(dim=1, keepdim=True)

    assert torch.allclose(graph_targets, expected)


def test_comatch_contrastive_loss_matches_graph_targets() -> None:
    feats_0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    feats_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    graph_targets = torch.eye(2, dtype=torch.float32)

    loss = comatch_contrastive_loss(
        feats_x_ulb_s_0=feats_0,
        feats_x_ulb_s_1=feats_1,
        graph_targets=graph_targets,
        temperature=0.5,
    )

    sim = torch.exp(feats_0 @ feats_1.t() / 0.5)
    sim_probs = sim / sim.sum(dim=1, keepdim=True)
    expected = -(torch.log(sim_probs + 1e-7) * graph_targets).sum(dim=1).mean()

    assert torch.allclose(loss, expected)


def test_comatch_memory_bank_updates_pointer_and_wraps() -> None:
    bank = CoMatchMemoryBank(queue_size=3, feature_dim=2, num_classes=2)

    bank.update(
        features=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        probabilities=torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32),
    )
    bank.update(
        features=torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
        probabilities=torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32),
    )

    assert bank.queue_ptr == 1
    assert bank.filled_size == 3
    assert torch.allclose(
        bank.feature_queue,
        torch.tensor([[0.0, 2.0], [0.0, 1.0], [2.0, 0.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        bank.probability_queue,
        torch.tensor([[0.4, 0.6], [0.2, 0.8], [0.7, 0.3]], dtype=torch.float32),
    )


def test_comatch_memory_bank_smooths_probabilities_from_active_queue() -> None:
    bank = CoMatchMemoryBank(queue_size=2, feature_dim=2, num_classes=2)
    bank.update(
        features=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        probabilities=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
    )
    features = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    probabilities = torch.tensor([[0.6, 0.4]], dtype=torch.float32)

    smoothed = bank.smooth_probabilities(
        features=features,
        probabilities=probabilities,
        temperature=1.0,
        smoothing_alpha=0.5,
    )

    affinities = torch.exp(features @ bank.active_features().t())
    affinities = affinities / affinities.sum(dim=1, keepdim=True)
    expected = 0.5 * probabilities + 0.5 * (affinities @ bank.active_probabilities())

    assert torch.allclose(smoothed, expected)


def test_comatch_memory_bank_roundtrips_state() -> None:
    bank = CoMatchMemoryBank(queue_size=2, feature_dim=2, num_classes=2)
    bank.update(
        features=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        probabilities=torch.tensor([[0.9, 0.1]], dtype=torch.float32),
    )
    state = bank.export_state()

    restored = CoMatchMemoryBank(queue_size=2, feature_dim=2, num_classes=2)
    restored.load_state(state)

    assert restored.queue_ptr == bank.queue_ptr
    assert restored.filled_size == bank.filled_size
    assert torch.allclose(restored.feature_queue, bank.feature_queue)
    assert torch.allclose(restored.probability_queue, bank.probability_queue)


def test_compute_comatch_step_updates_memory_bank_and_reports_losses() -> None:
    model = _FeatureMapModel(
        logits_by_token={
            1: torch.tensor([3.0, 0.0], dtype=torch.float32),
            2: torch.tensor([0.0, 3.0], dtype=torch.float32),
            3: torch.tensor([2.5, 0.0], dtype=torch.float32),
            4: torch.tensor([2.0, 0.0], dtype=torch.float32),
            5: torch.tensor([0.0, 2.0], dtype=torch.float32),
        },
        features_by_token={
            1: torch.tensor([1.0, 0.0], dtype=torch.float32),
            2: torch.tensor([0.0, 1.0], dtype=torch.float32),
            3: torch.tensor([1.0, 0.0], dtype=torch.float32),
            4: torch.tensor([1.0, 0.0], dtype=torch.float32),
            5: torch.tensor([0.0, 1.0], dtype=torch.float32),
        },
    )
    labeled_batch = {
        "input_ids": torch.tensor([[1], [2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1], [1]], dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    unlabeled_batch = {
        "weak_input_ids": torch.tensor([[3]], dtype=torch.long),
        "weak_attention_mask": torch.tensor([[1]], dtype=torch.long),
        "strong_0_input_ids": torch.tensor([[4]], dtype=torch.long),
        "strong_0_attention_mask": torch.tensor([[1]], dtype=torch.long),
        "strong_1_input_ids": torch.tensor([[5]], dtype=torch.long),
        "strong_1_attention_mask": torch.tensor([[1]], dtype=torch.long),
    }
    memory_bank = CoMatchMemoryBank(queue_size=4, feature_dim=2, num_classes=2)
    projection_head = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        projection_head.weight.copy_(torch.eye(2))

    output = compute_comatch_step(
        model=model,
        projection_head=projection_head,
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        dist_align_hook=QueueDistributionAlignmentHook(
            num_classes=2,
            queue_length=2,
        ),
        memory_bank=memory_bank,
        temperature=0.5,
        p_cutoff=0.1,
        contrast_p_cutoff=0.0,
        smoothing_alpha=0.9,
        lambda_u=1.0,
        lambda_c=1.0,
        supervised_loss_weight=1.0,
    )

    assert set(output.loss_components) == {
        "sup_loss",
        "unsup_loss",
        "contrast_loss",
    }
    assert output.total_loss.requires_grad
    assert memory_bank.filled_size == 3
    assert memory_bank.queue_ptr == 3
    assert output.metrics["util_ratio"].item() == 1.0


def test_comatch_algorithm_state_roundtrips_memory_bank() -> None:
    algorithm = CoMatchAlgorithm(
        temperature=0.5,
        p_cutoff=0.95,
        contrast_p_cutoff=0.8,
        queue_batch=4,
        smoothing_alpha=0.9,
        da_len=2,
        proj_size=2,
    )
    model = _FeatureMapModel(
        logits_by_token={1: torch.tensor([1.0, 0.0])},
        features_by_token={1: torch.tensor([1.0, 0.0])},
    )
    algorithm.configure_dataset(num_classes=2, unlabeled_row_count=8)
    algorithm.build_auxiliary_modules(model=model)
    assert algorithm.memory_bank is not None
    algorithm.memory_bank.update(
        features=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        probabilities=torch.tensor([[0.9, 0.1]], dtype=torch.float32),
    )
    state = algorithm.export_state()

    restored = CoMatchAlgorithm(
        temperature=0.5,
        p_cutoff=0.95,
        contrast_p_cutoff=0.8,
        queue_batch=4,
        smoothing_alpha=0.9,
        da_len=2,
        proj_size=2,
    )
    restored.configure_dataset(num_classes=2, unlabeled_row_count=8)
    restored.build_auxiliary_modules(model=model)
    restored.load_state(state)

    assert restored.memory_bank is not None
    assert restored.memory_bank.filled_size == 1
    assert torch.allclose(
        restored.memory_bank.feature_queue,
        algorithm.memory_bank.feature_queue,
    )
