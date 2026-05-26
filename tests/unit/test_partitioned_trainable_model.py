"""Physical trainable-adapter partition primitive tests."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from methods.adaptation.lora_classifier.federated_ssl import (
    partitioned_trainable_model as ptm,
)


class TinyFrozenFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(3))

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        del attention_mask
        return self.projection(input_ids.float())


def test_partition_plan_rejects_duplicate_partition_names() -> None:
    try:
        ptm.TrainableAdapterPartitionPlan.from_names(["sigma", "sigma"])
    except ValueError as error:
        assert "duplicates" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("duplicate partition names must fail")


def test_partition_plan_rejects_unknown_composition_policy() -> None:
    try:
        ptm.TrainableAdapterPartitionPlan.from_names(
            ["sigma", "psi"],
            composition_policy="merge_tensors",
        )
    except ValueError as error:
        assert "Unsupported composition policy" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("unknown composition policy must fail")


def test_physical_partition_optimizer_updates_only_selected_partition() -> None:
    model = _build_partitioned_model()
    sigma_before = ptm.snapshot_partition_parameters(model, "sigma")
    psi_before = ptm.snapshot_partition_parameters(model, "psi")
    feature_before = {
        name: parameter.detach().clone()
        for name, parameter in model.feature_extractor.named_parameters()
    }
    optimizer = torch.optim.SGD(model.partition_parameters("sigma"), lr=0.2)

    optimizer.zero_grad(set_to_none=True)
    logits = model.forward_partition(
        "sigma",
        input_ids=torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]),
        attention_mask=torch.ones(2, 3),
    )
    loss = F.cross_entropy(logits, torch.tensor([0, 1], dtype=torch.long))
    loss.backward()
    optimizer.step()

    sigma_after = ptm.snapshot_partition_parameters(model, "sigma")
    psi_after = ptm.snapshot_partition_parameters(model, "psi")
    feature_after = {
        name: parameter.detach().clone()
        for name, parameter in model.feature_extractor.named_parameters()
    }
    assert ptm.parameters_changed(before=sigma_before, after=sigma_after)
    assert not ptm.parameters_changed(before=psi_before, after=psi_after)
    assert not ptm.parameters_changed(before=feature_before, after=feature_after)


def test_partition_parameter_tensors_use_partition_local_names() -> None:
    model = _build_partitioned_model()

    parameter_tensors = model.partition_parameter_tensors("sigma")
    snapshot = ptm.snapshot_partition_parameter_tensors(model, "sigma")

    assert set(parameter_tensors) == {
        "adapter.weight",
        "classifier.weight",
        "classifier.bias",
    }
    assert set(snapshot) == set(parameter_tensors)
    assert all(not name.startswith("sigma.") for name in snapshot)


def test_composed_forward_sums_partition_logits() -> None:
    model = _build_partitioned_model()
    batch = {
        "input_ids": torch.tensor([[0.2, 0.4, 1.0]]),
        "attention_mask": torch.ones(1, 3),
    }

    sigma_logits = model.forward_partition("sigma", **batch)
    psi_logits = model.forward_partition("psi", **batch)
    composed_logits = model.forward_composed(**batch)

    torch.testing.assert_close(composed_logits, sigma_logits + psi_logits)


def _build_partitioned_model() -> ptm.PartitionedTrainableAdapterClassifier:
    return ptm.PartitionedTrainableAdapterClassifier(
        feature_extractor=TinyFrozenFeatureExtractor(),
        partitions=(
            _partition_spec("sigma", weight_scale=1.0),
            _partition_spec("psi", weight_scale=0.5),
        ),
    )


def _partition_spec(
    partition_name: str,
    *,
    weight_scale: float,
) -> ptm.AdapterClassifierPartitionSpec:
    adapter = nn.Linear(3, 3, bias=False)
    classifier = nn.Linear(3, 2)
    with torch.no_grad():
        adapter.weight.copy_(torch.eye(3) * weight_scale)
        classifier.weight.copy_(
            torch.tensor(
                [
                    [0.3, -0.2, 0.1],
                    [-0.1, 0.2, 0.4],
                ]
            )
        )
        classifier.bias.copy_(torch.tensor([0.05, -0.05]))
    return ptm.AdapterClassifierPartitionSpec(
        partition_name=partition_name,
        adapter=adapter,
        classifier=classifier,
    )
