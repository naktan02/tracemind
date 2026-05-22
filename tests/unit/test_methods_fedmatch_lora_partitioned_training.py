"""FedMatch LoRA-classifier partitioned optimizer step tests."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from methods.adaptation.lora_classifier.federated_ssl.fedmatch_partitioned_loop import (
    run_fedmatch_lora_classifier_partitioned_step,
    train_fedmatch_lora_classifier,
)
from methods.adaptation.lora_classifier.federated_ssl.fedmatch_training import (
    run_method_owned_lora_classifier_training_core,
)
from methods.adaptation.lora_classifier.federated_ssl.method_owned_training import (
    resolve_method_owned_lora_classifier_training_core,
)
from methods.adaptation.lora_classifier.training.partitioned_deltas import (
    build_lora_classifier_partition_delta_from_parameter_deltas,
    diff_parameter_snapshots,
    snapshot_trainable_parameter_tensors,
)
from methods.adaptation.query_classifier_adaptation.local_training_budget import (
    build_query_ssl_local_step_plan,
)
from methods.federated_ssl.fedmatch.local_objective import (
    FedMatchLocalObjectiveParameters,
)
from methods.federated_ssl.fedmatch.parameter_routing import (
    FEDMATCH_PSI_PARTITION,
    FEDMATCH_SIGMA_PARTITION,
)
from methods.ssl.algorithms.fixmatch.fixmatch import FixMatchAlgorithm


class TinyLoraClassifier(nn.Module):
    """transformer 의존 없이 LoRA/head parameter naming만 흉내내는 test model."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder_lora = nn.Linear(3, 3, bias=False)
        self.classifier = nn.Linear(3, 2)
        with torch.no_grad():
            self.encoder_lora.weight.copy_(
                torch.tensor(
                    [
                        [0.2, 0.0, 0.1],
                        [0.0, 0.3, 0.1],
                        [0.1, 0.0, 0.4],
                    ]
                )
            )
            self.classifier.weight.copy_(
                torch.tensor([[0.5, -0.2, 0.1], [-0.1, 0.3, 0.2]])
            )
            self.classifier.bias.copy_(torch.tensor([0.1, -0.1]))

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        del attention_mask
        return self.classifier(self.encoder_lora(input_ids.float()))


def test_method_owned_lora_classifier_core_resolves_descriptor_entrypoint() -> None:
    core = resolve_method_owned_lora_classifier_training_core("fedmatch")

    assert core is run_method_owned_lora_classifier_training_core


def test_lora_classifier_parameter_delta_can_be_split_into_partitions() -> None:
    partition = build_lora_classifier_partition_delta_from_parameter_deltas(
        partition_name="custom",
        labels=("anxiety", "normal"),
        parameter_deltas={
            "backbone.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "classifier.weight": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "classifier.bias": torch.tensor([0.5, -0.25]),
        },
    )

    assert partition.partition_name == "custom"
    assert partition.lora_parameter_deltas == {"backbone.lora_A.weight": [1.0, 2.0]}
    assert partition.classifier_head_weight_deltas == {
        "anxiety": [0.10000000149011612, 0.20000000298023224],
        "normal": [0.30000001192092896, 0.4000000059604645],
    }
    assert partition.classifier_head_bias_deltas == {
        "anxiety": 0.5,
        "normal": -0.25,
    }


def test_fedmatch_lora_partitioned_step_records_sigma_then_psi_delta() -> None:
    model = TinyLoraClassifier()
    labels = ("anxiety", "normal")
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.0,
        lambda_s=1.0,
        lambda_i=0.0,
        lambda_a=1.0,
        lambda_l2=0.0,
        lambda_l1=0.0,
    )
    sigma_optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    psi_optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    labeled_batch = {
        "input_ids": torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]),
        "attention_mask": torch.ones(2, 3),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    unlabeled_batch = {
        "weak_input_ids": torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.5, 1.0]]),
        "weak_attention_mask": torch.ones(2, 3),
        "strong_input_ids": torch.tensor([[0.8, 0.3, 0.1], [0.1, 0.4, 1.0]]),
        "strong_attention_mask": torch.ones(2, 3),
    }
    before = snapshot_trainable_parameter_tensors(model)

    result = run_fedmatch_lora_classifier_partitioned_step(
        model=model,
        labels=labels,
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        parameters=parameters,
        sigma_optimizer=sigma_optimizer,
        psi_optimizer=psi_optimizer,
    )

    after = snapshot_trainable_parameter_tensors(model)
    total_delta = diff_parameter_snapshots(after=after, before=before)
    assert set(result.partition_deltas) == {
        FEDMATCH_SIGMA_PARTITION,
        FEDMATCH_PSI_PARTITION,
    }
    assert "encoder_lora.weight" in (
        result.partition_deltas[FEDMATCH_SIGMA_PARTITION].lora_parameter_deltas
    )
    assert set(
        result.partition_deltas[
            FEDMATCH_SIGMA_PARTITION
        ].classifier_head_weight_deltas.keys()
    ) == set(labels)
    assert set(
        result.partition_deltas[
            FEDMATCH_PSI_PARTITION
        ].classifier_head_bias_deltas.keys()
    ) == set(labels)
    for name, delta in total_delta.items():
        torch.testing.assert_close(
            result.sigma_parameter_deltas[name] + result.psi_parameter_deltas[name],
            delta,
        )
    assert float(result.supervised.total_loss.detach().item()) > 0.0
    assert float(result.unsupervised.total_loss.detach().item()) > 0.0
    torch.testing.assert_close(
        result.metrics["sigma_labeled_count"],
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        result.metrics["psi_confident_count"],
        torch.tensor(2.0),
    )


def test_partitioned_step_can_use_fixmatch_for_psi_objective() -> None:
    model = TinyLoraClassifier()
    labels = ("anxiety", "normal")
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.0,
        lambda_s=1.0,
        lambda_i=0.0,
        lambda_a=1.0,
        lambda_l2=0.0,
        lambda_l1=0.0,
    )
    sigma_optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    psi_optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    labeled_batch = {
        "input_ids": torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]),
        "attention_mask": torch.ones(2, 3),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    unlabeled_batch = {
        "weak_input_ids": torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.5, 1.0]]),
        "weak_attention_mask": torch.ones(2, 3),
        "strong_input_ids": torch.tensor([[0.8, 0.3, 0.1], [0.1, 0.4, 1.0]]),
        "strong_attention_mask": torch.ones(2, 3),
    }

    result = run_fedmatch_lora_classifier_partitioned_step(
        model=model,
        labels=labels,
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        parameters=parameters,
        sigma_optimizer=sigma_optimizer,
        psi_optimizer=psi_optimizer,
        psi_query_ssl_algorithm=FixMatchAlgorithm(
            temperature=0.5,
            p_cutoff=0.0,
            hard_label=True,
            lambda_u=1.0,
            supervised_loss_weight=1.0,
        ),
    )

    assert set(result.partition_deltas) == {
        FEDMATCH_SIGMA_PARTITION,
        FEDMATCH_PSI_PARTITION,
    }
    assert "unsup_loss" in result.unsupervised.loss_components
    assert "util_ratio" in result.unsupervised.metrics


def test_fedmatch_lora_training_returns_cumulative_partitioned_delta() -> None:
    model = TinyLoraClassifier()
    labels = ("anxiety", "normal")
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.0,
        lambda_s=1.0,
        lambda_i=0.0,
        lambda_a=1.0,
        lambda_l2=0.0,
        lambda_l1=0.0,
    )
    train_loader = DataLoader(
        [
            {
                "input_ids": torch.tensor([1.0, 0.0, 0.5]),
                "attention_mask": torch.ones(3),
                "labels": torch.tensor(0, dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([0.0, 1.0, 0.5]),
                "attention_mask": torch.ones(3),
                "labels": torch.tensor(1, dtype=torch.long),
            },
        ],
        batch_size=2,
    )
    unlabeled_loader = DataLoader(
        [
            {
                "weak_input_ids": torch.tensor([1.0, 0.2, 0.0]),
                "weak_attention_mask": torch.ones(3),
                "strong_input_ids": torch.tensor([0.8, 0.3, 0.1]),
                "strong_attention_mask": torch.ones(3),
            },
            {
                "weak_input_ids": torch.tensor([0.0, 0.5, 1.0]),
                "weak_attention_mask": torch.ones(3),
                "strong_input_ids": torch.tensor([0.1, 0.4, 1.0]),
                "strong_attention_mask": torch.ones(3),
            },
        ],
        batch_size=2,
    )
    step_plan = build_query_ssl_local_step_plan(
        labeled_loader_steps=1,
        unlabeled_loader_steps=1,
        uses_labeled_batches=True,
        local_epochs=1,
        max_steps=2,
    )
    before = snapshot_trainable_parameter_tensors(model)

    result = train_fedmatch_lora_classifier(
        model=model,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        labels=labels,
        parameters=parameters,
        step_plan=step_plan,
        device="cpu",
        learning_rate=0.2,
        classifier_learning_rate=0.2,
        weight_decay=0.0,
        max_grad_norm=0.0,
    )

    after = snapshot_trainable_parameter_tensors(model)
    total_partition = build_lora_classifier_partition_delta_from_parameter_deltas(
        partition_name="total",
        labels=labels,
        parameter_deltas=diff_parameter_snapshots(after=after, before=before),
    )
    sigma = result.partition_deltas[FEDMATCH_SIGMA_PARTITION]
    psi = result.partition_deltas[FEDMATCH_PSI_PARTITION]
    combined_lora = [
        left + right
        for left, right in zip(
            sigma.lora_parameter_deltas["encoder_lora.weight"],
            psi.lora_parameter_deltas["encoder_lora.weight"],
        )
    ]

    assert result.metrics["train_total_loss"] > 0.0
    assert set(result.partition_deltas) == {
        FEDMATCH_SIGMA_PARTITION,
        FEDMATCH_PSI_PARTITION,
    }
    assert combined_lora == pytest.approx(
        total_partition.lora_parameter_deltas["encoder_lora.weight"]
    )
