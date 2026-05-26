"""LoRA-classifier training primitive characterization tests."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from methods.adaptation.local_objective_regularizers.fedprox import (
    prepare_fedprox_regularizer,
)
from methods.adaptation.text_classifier.peft_encoder.training.batching import (
    move_tensor_batch_to_device,
    next_cycling_batch,
)
from methods.adaptation.text_classifier.peft_encoder.training.optimizer_step import (
    run_optimizer_loss_step,
)
from methods.adaptation.text_classifier.peft_encoder.training.scalar_metrics import (
    ScalarMetricAccumulator,
    tensor_mapping_l2,
)
from methods.adaptation.text_classifier.peft_encoder.training.step_budget import (
    remaining_effective_epochs,
    resolve_epoch_distributed_step_budget,
)


def test_epoch_distributed_step_budget_preserves_max_step_formula() -> None:
    budget = resolve_epoch_distributed_step_budget(
        epochs=5,
        full_epoch_steps=10,
        max_train_steps=12,
        invalid_max_steps_message="invalid",
    )

    assert budget.total_train_steps == 12
    assert budget.steps_per_epoch_budget == 3
    assert budget.effective_epochs == 5
    assert budget.remaining_epoch_steps(completed_steps=9) == 3


def test_epoch_distributed_step_budget_preserves_epoch_formula_without_cap() -> None:
    budget = resolve_epoch_distributed_step_budget(
        epochs=2,
        full_epoch_steps=7,
        max_train_steps=None,
        invalid_max_steps_message="invalid",
    )

    assert budget.total_train_steps == 14
    assert budget.steps_per_epoch_budget == 7
    assert budget.effective_epochs == 2


def test_remaining_effective_epochs_preserves_resume_formula() -> None:
    assert (
        remaining_effective_epochs(
            epochs=5,
            remaining_steps=4,
            steps_per_epoch_budget=3,
        )
        == 2
    )


def test_next_cycling_batch_restarts_loader_after_stop_iteration() -> None:
    loader = DataLoader([{"value": torch.tensor(1)}], batch_size=1)
    iterator = iter(loader)

    first, iterator = next_cycling_batch(loader=loader, iterator=iterator)
    second, _iterator = next_cycling_batch(loader=loader, iterator=iterator)

    assert first["value"].item() == 1
    assert second["value"].item() == 1


def test_move_tensor_batch_to_device_preserves_metadata() -> None:
    moved = move_tensor_batch_to_device(
        batch={"input_ids": torch.tensor([1]), "row_id": "row-1"},
        device="cpu",
    )

    assert moved["input_ids"].device.type == "cpu"
    assert moved["row_id"] == "row-1"


def test_run_optimizer_loss_step_updates_parameters() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()

    loss = run_optimizer_loss_step(
        optimizer=optimizer,
        trainable_parameters=tuple(model.parameters()),
        max_grad_norm=1.0,
        compute_loss=lambda: model(torch.tensor([[1.0]])).sum(),
    )

    assert loss.detach().shape == torch.Size([])
    assert not torch.equal(model.weight.detach(), before)


def test_scalar_metric_accumulator_preserves_average_record_keys() -> None:
    accumulator = ScalarMetricAccumulator()
    accumulator.add_tensor("loss", torch.tensor(1.0))
    accumulator.add_float("loss", 2.0)
    accumulator.add_tensor_mapping({"util_ratio": torch.tensor(0.5)}, prefix="psi_")

    assert accumulator.average_record(denominator=2, key_prefix="train_") == {
        "train_loss": 1.5,
        "train_psi_util_ratio": 0.25,
    }
    assert accumulator.running_fields(denominator=2, key_prefix="running_") == [
        "running_loss=1.5000",
        "running_psi_util_ratio=0.2500",
    ]


def test_tensor_mapping_l2_matches_partition_delta_metric() -> None:
    assert tensor_mapping_l2({"a": torch.tensor([3.0, 4.0])}) == 5.0


def test_lora_training_fedprox_adds_loss_only_when_enabled() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    disabled = prepare_fedprox_regularizer(
        proximal_mu=0.0,
        trainable_parameters=(parameter,),
    )
    enabled = prepare_fedprox_regularizer(
        proximal_mu=0.5,
        trainable_parameters=(parameter,),
    )

    with torch.no_grad():
        parameter.add_(1.0)

    base_loss = parameter.new_tensor(2.0)
    assert disabled.add_to_loss(base_loss).item() == 2.0
    assert enabled.add_to_loss(base_loss).item() == 2.25
