"""CoMatch method-local primitive tests."""

from __future__ import annotations

import torch

from methods.ssl.algorithms.comatch.memory_bank import CoMatchMemoryBank


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
