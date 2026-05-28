"""FedMatch PEFT encoder partitioned optimizer step tests."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from methods.adaptation.peft_text_encoder.federated_ssl import (
    method_owned_training,
    peer_predictions,
)
from methods.adaptation.peft_text_encoder.federated_ssl.partitioned import (
    sparse_sync,
    training_loop,
)
from methods.adaptation.peft_text_encoder.federated_ssl.partitioned import (
    trainable_model as ptm,
)
from methods.adaptation.peft_text_encoder.training import (
    partitioned_deltas,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.adaptation.peft_text_encoder.update.partitioned_delta import (
    PeftEncoderPartitionDelta,
)
from methods.adaptation.query_text_views.local_training_budget import (
    build_query_ssl_local_step_plan,
)
from methods.federated_ssl.fedmatch.local_objective import (
    FEDMATCH_PSI_L1_REGULARIZATION,
    FEDMATCH_SIGMA_PSI_L2_REGULARIZATION,
    FedMatchLocalObjectiveParameters,
    build_fedmatch_partitioned_tensor_objective,
)
from methods.federated_ssl.fedmatch.parameter_routing import (
    FEDMATCH_PSI_PARTITION,
    FEDMATCH_SIGMA_PARTITION,
)
from methods.federated_ssl.fedmatch.partitioned_local_training import (
    run_method_owned_peft_encoder_training_core as run_fedmatch_partitioned_training,
)
from methods.ssl.algorithms.fixmatch.fixmatch import FixMatchAlgorithm

build_adapter_classifier_delta_bundle = (
    partitioned_deltas.build_adapter_classifier_delta_bundle
)
build_peft_encoder_partition_delta_from_parameter_deltas = (
    partitioned_deltas.build_peft_encoder_partition_delta_from_parameter_deltas
)
diff_parameter_snapshots = partitioned_deltas.diff_parameter_snapshots
project_adapter_classifier_delta_bundle_to_peft_partition_delta = (
    partitioned_deltas.project_adapter_classifier_delta_bundle_to_peft_partition_delta
)
snapshot_trainable_parameter_tensors = (
    partitioned_deltas.snapshot_trainable_parameter_tensors
)


def _fedmatch_objective(
    parameters: FedMatchLocalObjectiveParameters,
    *,
    single_model: bool = False,
):
    return build_fedmatch_partitioned_tensor_objective(
        parameters,
        omit_regularization_for_single_trainable_model=single_model,
    )


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


class TinyMismatchedPartitionClassifier(nn.Module):
    """sigma/psi trainable key mismatch를 드러내는 test model."""

    def __init__(self) -> None:
        super().__init__()
        self.other_adapter = nn.Linear(3, 3, bias=False)
        self.classifier = nn.Linear(3, 2)

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        del attention_mask
        return self.classifier(self.other_adapter(input_ids.float()))


class TinyLinearLogitClassifier(nn.Module):
    """composed sigma+psi confidence path를 분리해서 검증하는 test model."""

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.logits = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.logits.weight.copy_(weight)

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        del attention_mask
        return self.logits(input_ids.float())


class TinyFrozenFeatureExtractor(nn.Module):
    """물리 partition 테스트용 frozen backbone."""

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


def test_method_owned_peft_classifier_core_resolves_descriptor_entrypoint() -> None:
    core = method_owned_training.resolve_method_owned_peft_encoder_training_core(
        "fedmatch"
    )

    assert core is run_fedmatch_partitioned_training


def test_peft_encoder_parameter_delta_can_be_split_into_partitions() -> None:
    partition = build_peft_encoder_partition_delta_from_parameter_deltas(
        partition_name="custom",
        labels=("anxiety", "normal"),
        parameter_deltas={
            "backbone.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "classifier.weight": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "classifier.bias": torch.tensor([0.5, -0.25]),
        },
    )

    assert partition.partition_name == "custom"
    assert partition.peft_parameter_deltas == {"backbone.lora_A.weight": [1.0, 2.0]}
    assert partition.classifier_head_weight_deltas == {
        "anxiety": [0.10000000149011612, 0.20000000298023224],
        "normal": [0.30000001192092896, 0.4000000059604645],
    }
    assert partition.classifier_head_bias_deltas == {
        "anxiety": 0.5,
        "normal": -0.25,
    }


def test_peft_encoder_peer_snapshot_extracts_current_trainable_state() -> None:
    model = TinyLoraClassifier()
    labels = ("anxiety", "normal")

    snapshot = peer_predictions.extract_peft_encoder_materialized_state(
        model=model,  # type: ignore[arg-type]
        labels=labels,
    )

    assert "encoder_lora.weight" in snapshot.peft_parameters
    assert set(snapshot.classifier_head_weights) == set(labels)
    assert set(snapshot.classifier_head_biases) == set(labels)


def test_fedmatch_peft_encoder_partitioned_step_records_sigma_then_psi_delta() -> None:
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

    result = training_loop.run_partitioned_adapter_classifier_step(
        model=model,
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        objective=_fedmatch_objective(parameters, single_model=True),
        sigma_optimizer=sigma_optimizer,
        psi_optimizer=psi_optimizer,
    )

    after = snapshot_trainable_parameter_tensors(model)
    total_delta = diff_parameter_snapshots(after=after, before=before)
    sigma_delta = build_peft_encoder_partition_delta_from_parameter_deltas(
        partition_name=FEDMATCH_SIGMA_PARTITION,
        parameter_deltas=result.sigma_parameter_deltas,
        labels=labels,
    )
    psi_delta = build_peft_encoder_partition_delta_from_parameter_deltas(
        partition_name=FEDMATCH_PSI_PARTITION,
        parameter_deltas=result.psi_parameter_deltas,
        labels=labels,
    )
    assert "encoder_lora.weight" in sigma_delta.peft_parameter_deltas
    assert set(sigma_delta.classifier_head_weight_deltas) == set(labels)
    assert set(psi_delta.classifier_head_bias_deltas) == set(labels)
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


def test_adapter_classifier_delta_bundle_projects_to_peft_partition_delta() -> None:
    parameter_deltas = {
        "encoder_lora.weight": torch.tensor([[0.1, -0.2], [0.3, 0.4]]),
        "classifier.weight": torch.tensor([[0.5, 0.6], [-0.1, 0.2]]),
        "classifier.bias": torch.tensor([0.05, -0.07]),
    }

    bundle = build_adapter_classifier_delta_bundle(
        partition_name=FEDMATCH_SIGMA_PARTITION,
        parameter_deltas=parameter_deltas,
        labels=("anxiety", "normal"),
    )
    payload = project_adapter_classifier_delta_bundle_to_peft_partition_delta(
        bundle=bundle,
        labels=("anxiety", "normal"),
    )

    assert bundle.partition_name == FEDMATCH_SIGMA_PARTITION
    assert set(bundle.adapter_parameter_deltas) == {"encoder_lora.weight"}
    assert payload.partition_name == FEDMATCH_SIGMA_PARTITION
    assert payload.peft_parameter_deltas["encoder_lora.weight"] == pytest.approx(
        [0.1, -0.2, 0.3, 0.4]
    )
    assert payload.classifier_head_weight_deltas["anxiety"] == pytest.approx([0.5, 0.6])
    assert payload.classifier_head_bias_deltas["normal"] == pytest.approx(-0.07)


def test_partitioned_step_can_use_fixmatch_for_psi_objective() -> None:
    model = TinyLoraClassifier()
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

    result = training_loop.run_partitioned_adapter_classifier_step(
        model=model,
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        objective=_fedmatch_objective(parameters, single_model=True),
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

    assert set(result.sigma_parameter_deltas) == set(result.psi_parameter_deltas)
    assert "unsup_loss" in result.unsupervised.loss_components
    assert "util_ratio" in result.unsupervised.metrics


def test_fedmatch_partitioned_step_forwards_strong_view_only_for_confident_rows() -> (
    None
):
    class CountingTinyLoraClassifier(TinyLoraClassifier):
        def __init__(self) -> None:
            super().__init__()
            self.forward_shapes: list[tuple[int, int]] = []

        def forward(
            self,
            *,
            input_ids: Tensor,
            attention_mask: Tensor,
        ) -> Tensor:
            self.forward_shapes.append(tuple(input_ids.shape))
            logits = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if len(self.forward_shapes) == 2:
                return torch.tensor(
                    [[4.0, 0.0], [0.0, 4.0], [0.55, 0.45]],
                    dtype=logits.dtype,
                    device=logits.device,
                )
            return logits

    model = CountingTinyLoraClassifier()
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.9,
        lambda_s=1.0,
        lambda_i=0.0,
        lambda_a=1.0,
        lambda_l2=0.0,
        lambda_l1=0.0,
    )
    sigma_optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    psi_optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    result = training_loop.run_partitioned_adapter_classifier_step(
        model=model,
        labeled_batch={
            "input_ids": torch.tensor([[1.0, 0.0, 0.5]]),
            "attention_mask": torch.ones(1, 3),
            "labels": torch.tensor([0], dtype=torch.long),
        },
        unlabeled_batch={
            "weak_input_ids": torch.tensor(
                [[1.0, 0.2, 0.0], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]]
            ),
            "weak_attention_mask": torch.ones(3, 3),
            "strong_input_ids": torch.tensor(
                [[0.8, 0.3, 0.1], [0.1, 0.4, 1.0], [0.5, 0.5, 0.5]]
            ),
            "strong_attention_mask": torch.ones(3, 3),
        },
        objective=_fedmatch_objective(parameters, single_model=True),
        sigma_optimizer=sigma_optimizer,
        psi_optimizer=psi_optimizer,
    )

    assert model.forward_shapes == [(1, 3), (3, 3), (2, 3)]
    torch.testing.assert_close(
        result.unsupervised.metrics["confident_count"],
        torch.tensor(2.0),
    )


def test_fedmatch_partitioned_step_requests_helper_probs_only_for_confident_rows() -> (
    None
):
    class CountingTinyLoraClassifier(TinyLoraClassifier):
        def __init__(self) -> None:
            super().__init__()
            self.forward_shapes: list[tuple[int, int]] = []

        def forward(
            self,
            *,
            input_ids: Tensor,
            attention_mask: Tensor,
        ) -> Tensor:
            self.forward_shapes.append(tuple(input_ids.shape))
            logits = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if len(self.forward_shapes) == 2:
                return torch.tensor(
                    [[4.0, 0.0], [0.0, 4.0], [0.55, 0.45]],
                    dtype=logits.dtype,
                    device=logits.device,
                )
            return logits

    provider_calls: list[tuple[int, int]] = []

    def helper_provider(
        *,
        unlabeled_batch: dict[str, Tensor],
    ) -> Tensor:
        input_ids = unlabeled_batch["weak_input_ids"]
        provider_calls.append(tuple(input_ids.shape))
        torch.testing.assert_close(
            input_ids,
            torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.5, 1.0]]),
        )
        return torch.tensor(
            [[[0.9, 0.1], [0.1, 0.9]]],
            dtype=torch.float32,
        )

    model = CountingTinyLoraClassifier()
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.9,
        lambda_s=1.0,
        lambda_i=1.0,
        lambda_a=1.0,
        lambda_l2=0.0,
        lambda_l1=0.0,
    )
    sigma_optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    psi_optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    result = training_loop.run_partitioned_adapter_classifier_step(
        model=model,
        labeled_batch={
            "input_ids": torch.tensor([[1.0, 0.0, 0.5]]),
            "attention_mask": torch.ones(1, 3),
            "labels": torch.tensor([0], dtype=torch.long),
        },
        unlabeled_batch={
            "weak_input_ids": torch.tensor(
                [[1.0, 0.2, 0.0], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]]
            ),
            "weak_attention_mask": torch.ones(3, 3),
            "strong_input_ids": torch.tensor(
                [[0.8, 0.3, 0.1], [0.1, 0.4, 1.0], [0.5, 0.5, 0.5]]
            ),
            "strong_attention_mask": torch.ones(3, 3),
        },
        objective=_fedmatch_objective(parameters, single_model=True),
        sigma_optimizer=sigma_optimizer,
        psi_optimizer=psi_optimizer,
        helper_weak_probability_provider=helper_provider,
    )

    assert provider_calls == [(2, 3)]
    torch.testing.assert_close(
        result.unsupervised.metrics["confident_count"],
        torch.tensor(2.0),
    )


def test_fedmatch_peft_single_model_regularizer_does_not_shrink_full_parameters() -> (
    None
):
    model = TinyLoraClassifier()
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.99,
        lambda_s=1.0,
        lambda_i=0.0,
        lambda_a=1.0,
        lambda_l2=10.0,
        lambda_l1=1.0,
    )
    psi_optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    unlabeled_batch = {
        "weak_input_ids": torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.5, 1.0]]),
        "weak_attention_mask": torch.ones(2, 3),
        "strong_input_ids": torch.tensor([[0.8, 0.3, 0.1], [0.1, 0.4, 1.0]]),
        "strong_attention_mask": torch.ones(2, 3),
    }
    before = snapshot_trainable_parameter_tensors(model)

    result = training_loop.run_partitioned_adapter_classifier_step(
        model=model,
        labeled_batch=None,
        unlabeled_batch=unlabeled_batch,
        objective=_fedmatch_objective(parameters, single_model=True),
        sigma_optimizer=torch.optim.SGD(model.parameters(), lr=0.2),
        psi_optimizer=psi_optimizer,
        apply_supervised_step=False,
    )

    after = snapshot_trainable_parameter_tensors(model)
    for name, before_tensor in before.items():
        torch.testing.assert_close(after[name], before_tensor)
        torch.testing.assert_close(
            result.psi_parameter_deltas[name],
            torch.zeros_like(before_tensor),
        )
    torch.testing.assert_close(
        result.unsupervised.metrics["confident_count"],
        torch.tensor(0.0),
    )


def test_physical_fedmatch_step_updates_separate_sigma_and_psi_partitions() -> None:
    model = _build_physical_partitioned_model()
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.0,
        lambda_s=1.0,
        lambda_i=0.0,
        lambda_a=1.0,
        lambda_l2=0.0,
        lambda_l1=0.0,
    )
    feature_before = {
        name: parameter.detach().clone()
        for name, parameter in model.feature_extractor.named_parameters()
    }
    sigma_before = ptm.snapshot_partition_parameter_tensors(
        model,
        FEDMATCH_SIGMA_PARTITION,
    )
    psi_before = ptm.snapshot_partition_parameter_tensors(
        model,
        FEDMATCH_PSI_PARTITION,
    )

    result = training_loop.run_physical_partitioned_adapter_classifier_step(
        model=model,
        labeled_batch={
            "input_ids": torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]),
            "attention_mask": torch.ones(2, 3),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        },
        unlabeled_batch={
            "weak_input_ids": torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.5, 1.0]]),
            "weak_attention_mask": torch.ones(2, 3),
            "strong_input_ids": torch.tensor([[0.8, 0.3, 0.1], [0.1, 0.4, 1.0]]),
            "strong_attention_mask": torch.ones(2, 3),
        },
        objective=_fedmatch_objective(parameters),
        supervised_partition=FEDMATCH_SIGMA_PARTITION,
        unsupervised_partition=FEDMATCH_PSI_PARTITION,
        sigma_optimizer=torch.optim.SGD(
            model.partition_parameters(FEDMATCH_SIGMA_PARTITION),
            lr=0.2,
        ),
        psi_optimizer=torch.optim.SGD(
            model.partition_parameters(FEDMATCH_PSI_PARTITION),
            lr=0.2,
        ),
    )

    feature_after = {
        name: parameter.detach().clone()
        for name, parameter in model.feature_extractor.named_parameters()
    }
    sigma_after = ptm.snapshot_partition_parameter_tensors(
        model,
        FEDMATCH_SIGMA_PARTITION,
    )
    psi_after = ptm.snapshot_partition_parameter_tensors(
        model,
        FEDMATCH_PSI_PARTITION,
    )
    assert set(result.sigma_parameter_deltas) == {
        "adapter.weight",
        "classifier.weight",
        "classifier.bias",
    }
    assert set(result.psi_parameter_deltas) == set(result.sigma_parameter_deltas)
    assert ptm.parameters_changed(before=sigma_before, after=sigma_after)
    assert ptm.parameters_changed(before=psi_before, after=psi_after)
    assert not ptm.parameters_changed(before=feature_before, after=feature_after)
    torch.testing.assert_close(
        result.metrics["sigma_labeled_count"],
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        result.metrics["psi_confident_count"],
        torch.tensor(2.0),
    )


def test_physical_fedmatch_unsupervised_regularizer_keeps_sigma_fixed() -> None:
    model = _build_physical_partitioned_model()
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.99,
        lambda_s=1.0,
        lambda_i=0.0,
        lambda_a=1.0,
        lambda_l2=0.5,
        lambda_l1=0.1,
    )
    sigma_before = ptm.snapshot_partition_parameter_tensors(
        model,
        FEDMATCH_SIGMA_PARTITION,
    )
    psi_before = ptm.snapshot_partition_parameter_tensors(
        model,
        FEDMATCH_PSI_PARTITION,
    )

    result = training_loop.run_physical_partitioned_adapter_classifier_step(
        model=model,
        labeled_batch=None,
        unlabeled_batch={
            "weak_input_ids": torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.5, 1.0]]),
            "weak_attention_mask": torch.ones(2, 3),
            "strong_input_ids": torch.tensor([[0.8, 0.3, 0.1], [0.1, 0.4, 1.0]]),
            "strong_attention_mask": torch.ones(2, 3),
        },
        objective=_fedmatch_objective(parameters),
        supervised_partition=FEDMATCH_SIGMA_PARTITION,
        unsupervised_partition=FEDMATCH_PSI_PARTITION,
        sigma_optimizer=torch.optim.SGD(
            model.partition_parameters(FEDMATCH_SIGMA_PARTITION),
            lr=0.2,
        ),
        psi_optimizer=torch.optim.SGD(
            model.partition_parameters(FEDMATCH_PSI_PARTITION),
            lr=0.2,
        ),
        apply_supervised_step=False,
    )

    sigma_after = ptm.snapshot_partition_parameter_tensors(
        model,
        FEDMATCH_SIGMA_PARTITION,
    )
    psi_after = ptm.snapshot_partition_parameter_tensors(
        model,
        FEDMATCH_PSI_PARTITION,
    )
    assert not ptm.parameters_changed(before=sigma_before, after=sigma_after)
    assert ptm.parameters_changed(before=psi_before, after=psi_after)
    torch.testing.assert_close(
        result.unsupervised.metrics["confident_count"],
        torch.tensor(0.0),
    )
    assert (
        float(
            result.unsupervised.loss_components[FEDMATCH_PSI_L1_REGULARIZATION].detach()
        )
        > 0.0
    )
    assert (
        float(
            result.unsupervised.loss_components[
                FEDMATCH_SIGMA_PSI_L2_REGULARIZATION
            ].detach()
        )
        > 0.0
    )


def test_physical_fedmatch_training_returns_cumulative_partitioned_delta() -> None:
    model = _build_physical_partitioned_model()
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

    result = training_loop.train_physical_partitioned_adapter_classifier(
        model=model,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        labels=labels,
        objective=_fedmatch_objective(parameters),
        step_plan=step_plan,
        device="cpu",
        learning_rate=0.2,
        classifier_learning_rate=0.2,
        weight_decay=0.0,
        max_grad_norm=0.0,
        supervised_partition=FEDMATCH_SIGMA_PARTITION,
        unsupervised_partition=FEDMATCH_PSI_PARTITION,
        metric_prefix="fedmatch",
    )

    sigma = result.partition_deltas[FEDMATCH_SIGMA_PARTITION]
    psi = result.partition_deltas[FEDMATCH_PSI_PARTITION]
    assert result.metrics["train_total_loss"] > 0.0
    assert result.metrics["train_fedmatch_sigma_delta_l2"] > 0.0
    assert result.metrics["train_fedmatch_psi_delta_l2"] > 0.0
    assert set(sigma.peft_parameter_deltas) == {"adapter.weight"}
    assert set(psi.peft_parameter_deltas) == {"adapter.weight"}
    assert set(sigma.classifier_head_weight_deltas) == set(labels)
    assert set(psi.classifier_head_bias_deltas) == set(labels)


def test_physical_fedmatch_training_accepts_full_text_classifier_partitions() -> None:
    model = ptm.PartitionedTrainableTextClassifierModules(
        partitions=(
            ptm.TextClassifierPartitionSpec(
                partition_name=FEDMATCH_SIGMA_PARTITION,
                module=TinyLoraClassifier(),
            ),
            ptm.TextClassifierPartitionSpec(
                partition_name=FEDMATCH_PSI_PARTITION,
                module=TinyLoraClassifier(),
            ),
        )
    )
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
        max_steps=1,
    )

    result = training_loop.train_physical_partitioned_adapter_classifier(
        model=model,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        labels=labels,
        objective=_fedmatch_objective(parameters),
        step_plan=step_plan,
        device="cpu",
        learning_rate=0.2,
        classifier_learning_rate=0.2,
        weight_decay=0.0,
        max_grad_norm=0.0,
        supervised_partition=FEDMATCH_SIGMA_PARTITION,
        unsupervised_partition=FEDMATCH_PSI_PARTITION,
        metric_prefix="fedmatch",
    )

    sigma = result.partition_deltas[FEDMATCH_SIGMA_PARTITION]
    psi = result.partition_deltas[FEDMATCH_PSI_PARTITION]
    assert set(sigma.peft_parameter_deltas) == {"encoder_lora.weight"}
    assert set(psi.peft_parameter_deltas) == {"encoder_lora.weight"}
    assert set(sigma.classifier_head_weight_deltas) == set(labels)
    assert set(psi.classifier_head_bias_deltas) == set(labels)


def test_physical_fedmatch_confidence_uses_sigma_plus_psi_forward() -> None:
    model = ptm.PartitionedTrainableTextClassifierModules(
        partitions=(
            ptm.TextClassifierPartitionSpec(
                partition_name=FEDMATCH_SIGMA_PARTITION,
                module=TinyLinearLogitClassifier(
                    torch.tensor([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
                ),
            ),
            ptm.TextClassifierPartitionSpec(
                partition_name=FEDMATCH_PSI_PARTITION,
                module=TinyLinearLogitClassifier(torch.zeros(2, 3)),
            ),
        )
    )
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.9,
        lambda_s=1.0,
        lambda_i=0.0,
        lambda_a=1.0,
        lambda_l2=0.0,
        lambda_l1=0.0,
    )

    result = training_loop.run_physical_partitioned_adapter_classifier_step(
        model=model,
        labeled_batch=None,
        unlabeled_batch={
            "weak_input_ids": torch.tensor([[1.0, 0.0, 0.0]]),
            "weak_attention_mask": torch.ones(1, 3),
            "strong_input_ids": torch.tensor([[1.0, 0.0, 0.0]]),
            "strong_attention_mask": torch.ones(1, 3),
        },
        objective=_fedmatch_objective(parameters),
        supervised_partition=FEDMATCH_SIGMA_PARTITION,
        unsupervised_partition=FEDMATCH_PSI_PARTITION,
        sigma_optimizer=torch.optim.SGD(
            model.partition_parameters(FEDMATCH_SIGMA_PARTITION),
            lr=0.2,
        ),
        psi_optimizer=torch.optim.SGD(
            model.partition_parameters(FEDMATCH_PSI_PARTITION),
            lr=0.2,
        ),
        apply_supervised_step=False,
    )

    torch.testing.assert_close(
        result.unsupervised.metrics["confident_count"],
        torch.tensor(1.0),
    )
    assert (
        ptm.snapshot_partition_parameter_tensors(
            model,
            FEDMATCH_PSI_PARTITION,
        )["logits.weight"][0, 0]
        > 0.0
    )


def test_physical_fedmatch_full_text_partition_rejects_key_mismatch() -> None:
    model = ptm.PartitionedTrainableTextClassifierModules(
        partitions=(
            ptm.TextClassifierPartitionSpec(
                partition_name=FEDMATCH_SIGMA_PARTITION,
                module=TinyLoraClassifier(),
            ),
            ptm.TextClassifierPartitionSpec(
                partition_name=FEDMATCH_PSI_PARTITION,
                module=TinyMismatchedPartitionClassifier(),
            ),
        )
    )
    parameters = FedMatchLocalObjectiveParameters(
        confidence_threshold=0.0,
        lambda_s=1.0,
        lambda_i=0.0,
        lambda_a=1.0,
        lambda_l2=0.1,
        lambda_l1=0.0,
    )

    with pytest.raises(ValueError, match="same parameter keys"):
        training_loop.run_physical_partitioned_adapter_classifier_step(
            model=model,
            labeled_batch=None,
            unlabeled_batch={
                "weak_input_ids": torch.tensor([[1.0, 0.2, 0.0]]),
                "weak_attention_mask": torch.ones(1, 3),
                "strong_input_ids": torch.tensor([[0.8, 0.3, 0.1]]),
                "strong_attention_mask": torch.ones(1, 3),
            },
            objective=_fedmatch_objective(parameters),
            supervised_partition=FEDMATCH_SIGMA_PARTITION,
            unsupervised_partition=FEDMATCH_PSI_PARTITION,
            sigma_optimizer=torch.optim.SGD(
                model.partition_parameters(FEDMATCH_SIGMA_PARTITION),
                lr=0.2,
            ),
            psi_optimizer=torch.optim.SGD(
                model.partition_parameters(FEDMATCH_PSI_PARTITION),
                lr=0.2,
            ),
            apply_supervised_step=False,
        )


def test_fedmatch_peft_training_returns_cumulative_partitioned_delta() -> None:
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

    result = training_loop.train_partitioned_adapter_classifier(
        model=model,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        labels=labels,
        objective=_fedmatch_objective(parameters, single_model=True),
        step_plan=step_plan,
        device="cpu",
        learning_rate=0.2,
        classifier_learning_rate=0.2,
        weight_decay=0.0,
        max_grad_norm=0.0,
        supervised_partition=FEDMATCH_SIGMA_PARTITION,
        unsupervised_partition=FEDMATCH_PSI_PARTITION,
        metric_prefix="fedmatch",
    )

    after = snapshot_trainable_parameter_tensors(model)
    total_partition = build_peft_encoder_partition_delta_from_parameter_deltas(
        partition_name="total",
        labels=labels,
        parameter_deltas=diff_parameter_snapshots(after=after, before=before),
    )
    sigma = result.partition_deltas[FEDMATCH_SIGMA_PARTITION]
    psi = result.partition_deltas[FEDMATCH_PSI_PARTITION]
    combined_lora = [
        left + right
        for left, right in zip(
            sigma.peft_parameter_deltas["encoder_lora.weight"],
            psi.peft_parameter_deltas["encoder_lora.weight"],
        )
    ]

    assert result.metrics["train_total_loss"] > 0.0
    assert set(result.partition_deltas) == {
        FEDMATCH_SIGMA_PARTITION,
        FEDMATCH_PSI_PARTITION,
    }
    assert combined_lora == pytest.approx(
        total_partition.peft_parameter_deltas["encoder_lora.weight"]
    )


def test_fedmatch_labels_at_server_training_uploads_only_psi_partition() -> None:
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
        labeled_loader_steps=0,
        unlabeled_loader_steps=1,
        uses_labeled_batches=False,
        local_epochs=1,
        max_steps=1,
    )

    result = training_loop.train_partitioned_adapter_classifier(
        model=model,
        train_loader=None,
        unlabeled_loader=unlabeled_loader,
        labels=labels,
        objective=_fedmatch_objective(parameters, single_model=True),
        step_plan=step_plan,
        device="cpu",
        learning_rate=0.2,
        classifier_learning_rate=0.2,
        weight_decay=0.0,
        max_grad_norm=0.0,
        use_supervised_steps=False,
        supervised_partition=FEDMATCH_SIGMA_PARTITION,
        unsupervised_partition=FEDMATCH_PSI_PARTITION,
        emit_sigma_partition=False,
        metric_prefix="fedmatch",
    )

    assert set(result.partition_deltas) == {FEDMATCH_PSI_PARTITION}
    assert result.metrics["train_sup_loss"] == 0.0
    assert result.metrics["train_fedmatch_sigma_delta_l2"] == 0.0
    assert result.metrics["train_fedmatch_psi_delta_l2"] > 0.0


def test_partitioned_c2s_sparse_upload_cuts_delta_and_sparsifies_psi() -> None:
    base = PeftEncoderMaterializedState(
        peft_parameters={"encoder_lora.weight": [0.10, 0.10, 0.10]},
        classifier_head_weights={"anxiety": [0.10, 0.10]},
        classifier_head_biases={"anxiety": 0.10},
    )
    partition_base = {
        FEDMATCH_PSI_PARTITION: PeftEncoderMaterializedState(
            peft_parameters={"encoder_lora.weight": [0.03, 0.10, 0.10]},
            classifier_head_weights={"anxiety": [0.03, 0.10]},
            classifier_head_biases={"anxiety": 0.03},
        )
    }
    sparse = sparse_sync.apply_partitioned_c2s_sparse_upload(
        base_parameters=base,
        base_partition_parameters=partition_base,
        partition_deltas={
            FEDMATCH_SIGMA_PARTITION: PeftEncoderPartitionDelta(
                partition_name=FEDMATCH_SIGMA_PARTITION,
                peft_parameter_deltas={"encoder_lora.weight": [0.01, 0.04, -0.06]},
                classifier_head_weight_deltas={"anxiety": [0.02, 0.07]},
                classifier_head_bias_deltas={"anxiety": 0.03},
            ),
            FEDMATCH_PSI_PARTITION: PeftEncoderPartitionDelta(
                partition_name=FEDMATCH_PSI_PARTITION,
                peft_parameter_deltas={"encoder_lora.weight": [0.01, -0.01, 0.08]},
                classifier_head_weight_deltas={"anxiety": [0.01, -0.01]},
                classifier_head_bias_deltas={"anxiety": 0.01},
            ),
        },
        parameters=sparse_sync.PartitionSparseSyncParameters(
            l1_threshold=0.05,
            delta_threshold=0.02,
            l1_sparse_partitions=(FEDMATCH_PSI_PARTITION,),
        ),
    )

    assert sparse[FEDMATCH_SIGMA_PARTITION].peft_parameter_deltas[
        "encoder_lora.weight"
    ] == pytest.approx([0.0, 0.04, -0.06])
    assert sparse[FEDMATCH_SIGMA_PARTITION].classifier_head_weight_deltas[
        "anxiety"
    ] == pytest.approx([0.0, 0.07])
    assert sparse[FEDMATCH_PSI_PARTITION].peft_parameter_deltas[
        "encoder_lora.weight"
    ] == pytest.approx([-0.03, 0.0, 0.08])
    assert sparse[FEDMATCH_PSI_PARTITION].classifier_head_weight_deltas[
        "anxiety"
    ] == pytest.approx([-0.03, 0.0])
    assert sparse[FEDMATCH_PSI_PARTITION].classifier_head_bias_deltas[
        "anxiety"
    ] == pytest.approx(-0.03)


def test_partitioned_s2c_sparse_download_diffs_server_and_client_partitions() -> None:
    server_partitions = {
        FEDMATCH_SIGMA_PARTITION: PeftEncoderMaterializedState(
            peft_parameters={"encoder_lora.weight": [0.10, 0.14, 0.04]},
            classifier_head_weights={"anxiety": [0.10, 0.17]},
            classifier_head_biases={"anxiety": 0.12},
        ),
        FEDMATCH_PSI_PARTITION: PeftEncoderMaterializedState(
            peft_parameters={"encoder_lora.weight": [0.04, 0.10, 0.21]},
            classifier_head_weights={"anxiety": [0.04, 0.10]},
            classifier_head_biases={"anxiety": 0.04},
        ),
    }
    client_partitions = {
        FEDMATCH_SIGMA_PARTITION: PeftEncoderMaterializedState(
            peft_parameters={"encoder_lora.weight": [0.09, 0.10, 0.10]},
            classifier_head_weights={"anxiety": [0.085, 0.10]},
            classifier_head_biases={"anxiety": 0.09},
        ),
        FEDMATCH_PSI_PARTITION: PeftEncoderMaterializedState(
            peft_parameters={"encoder_lora.weight": [0.03, 0.085, 0.11]},
            classifier_head_weights={"anxiety": [0.03, 0.085]},
            classifier_head_biases={"anxiety": 0.03},
        ),
    }

    sparse = sparse_sync.apply_partitioned_s2c_sparse_download(
        server_partition_parameters=server_partitions,
        client_partition_parameters=client_partitions,
        parameters=sparse_sync.PartitionSparseSyncParameters(
            l1_threshold=0.05,
            delta_threshold=0.02,
            l1_sparse_partitions=(FEDMATCH_PSI_PARTITION,),
        ),
    )

    assert sparse[FEDMATCH_SIGMA_PARTITION].peft_parameter_deltas[
        "encoder_lora.weight"
    ] == pytest.approx([0.0, 0.04, -0.06])
    assert sparse[FEDMATCH_SIGMA_PARTITION].classifier_head_weight_deltas[
        "anxiety"
    ] == pytest.approx([0.0, 0.07])
    assert sparse[FEDMATCH_SIGMA_PARTITION].classifier_head_bias_deltas[
        "anxiety"
    ] == pytest.approx(0.03)
    assert sparse[FEDMATCH_PSI_PARTITION].peft_parameter_deltas[
        "encoder_lora.weight"
    ] == pytest.approx([0.0, 0.0, 0.10])
    assert sparse[FEDMATCH_PSI_PARTITION].classifier_head_weight_deltas[
        "anxiety"
    ] == pytest.approx([0.0, 0.0])
    assert sparse[FEDMATCH_PSI_PARTITION].classifier_head_bias_deltas[
        "anxiety"
    ] == pytest.approx(0.0)


def test_partition_delta_nonzero_count_tracks_sparse_transport_values() -> None:
    deltas = {
        FEDMATCH_SIGMA_PARTITION: PeftEncoderPartitionDelta(
            partition_name=FEDMATCH_SIGMA_PARTITION,
            peft_parameter_deltas={"encoder_lora.weight": [0.0, 0.04, -0.06]},
            classifier_head_weight_deltas={"anxiety": [0.0, 0.07]},
            classifier_head_bias_deltas={"anxiety": 0.03},
        ),
        FEDMATCH_PSI_PARTITION: PeftEncoderPartitionDelta(
            partition_name=FEDMATCH_PSI_PARTITION,
            peft_parameter_deltas={"encoder_lora.weight": [0.0, 0.0, 0.10]},
            classifier_head_weight_deltas={"anxiety": [0.0, 0.0]},
            classifier_head_bias_deltas={"anxiety": 0.0},
        ),
    }

    assert sparse_sync.count_partition_delta_nonzero_values(deltas) == 5


def test_partitioned_s2c_projection_keeps_raw_server_values_after_sparse_mask() -> None:
    server_partitions = {
        FEDMATCH_PSI_PARTITION: PeftEncoderMaterializedState(
            peft_parameters={"encoder_lora.weight": [0.04, 0.21]},
            classifier_head_weights={"anxiety": [0.04, 0.21]},
            classifier_head_biases={"anxiety": 0.04},
        )
    }
    client_partitions = {
        FEDMATCH_PSI_PARTITION: PeftEncoderMaterializedState(
            peft_parameters={"encoder_lora.weight": [0.09, 0.11]},
            classifier_head_weights={"anxiety": [0.09, 0.11]},
            classifier_head_biases={"anxiety": 0.09},
        )
    }

    projected = sparse_sync.project_partitioned_s2c_sparse_download(
        server_partition_parameters=server_partitions,
        client_partition_parameters=client_partitions,
        parameters=sparse_sync.PartitionSparseSyncParameters(
            l1_threshold=0.05,
            delta_threshold=0.02,
            l1_sparse_partitions=(FEDMATCH_PSI_PARTITION,),
        ),
    )
    sparse_delta = sparse_sync.apply_partitioned_s2c_sparse_download(
        server_partition_parameters=server_partitions,
        client_partition_parameters=client_partitions,
        parameters=sparse_sync.PartitionSparseSyncParameters(
            l1_threshold=0.05,
            delta_threshold=0.02,
            l1_sparse_partitions=(FEDMATCH_PSI_PARTITION,),
        ),
    )

    assert projected[FEDMATCH_PSI_PARTITION].peft_parameters[
        "encoder_lora.weight"
    ] == pytest.approx([0.04, 0.21])
    assert projected[FEDMATCH_PSI_PARTITION].classifier_head_weights[
        "anxiety"
    ] == pytest.approx([0.04, 0.21])
    assert projected[FEDMATCH_PSI_PARTITION].classifier_head_biases[
        "anxiety"
    ] == pytest.approx(0.04)
    assert sparse_delta[FEDMATCH_PSI_PARTITION].peft_parameter_deltas[
        "encoder_lora.weight"
    ] == pytest.approx([-0.05, 0.10])


def test_partitioned_c2s_projection_returns_post_upload_client_snapshot() -> None:
    server_partitions = {
        FEDMATCH_SIGMA_PARTITION: PeftEncoderMaterializedState(
            peft_parameters={"encoder_lora.weight": [0.10, 0.20]},
            classifier_head_weights={"anxiety": [0.10, 0.20]},
            classifier_head_biases={"anxiety": 0.10},
        ),
        FEDMATCH_PSI_PARTITION: PeftEncoderMaterializedState(
            peft_parameters={"encoder_lora.weight": [0.10, 0.20]},
            classifier_head_weights={"anxiety": [0.10, 0.20]},
            classifier_head_biases={"anxiety": 0.10},
        ),
    }
    client_partitions = {
        FEDMATCH_SIGMA_PARTITION: PeftEncoderMaterializedState(
            peft_parameters={"encoder_lora.weight": [0.11, 0.25]},
            classifier_head_weights={"anxiety": [0.11, 0.25]},
            classifier_head_biases={"anxiety": 0.11},
        ),
        FEDMATCH_PSI_PARTITION: PeftEncoderMaterializedState(
            peft_parameters={"encoder_lora.weight": [0.04, 0.30]},
            classifier_head_weights={"anxiety": [0.04, 0.30]},
            classifier_head_biases={"anxiety": 0.04},
        ),
    }

    projection = sparse_sync.project_partitioned_c2s_sparse_upload(
        base_parameters=PeftEncoderMaterializedState(
            peft_parameters={},
            classifier_head_weights={},
            classifier_head_biases={},
        ),
        server_partition_parameters=server_partitions,
        client_partition_parameters=client_partitions,
        parameters=sparse_sync.PartitionSparseSyncParameters(
            l1_threshold=0.05,
            delta_threshold=0.02,
            l1_sparse_partitions=(FEDMATCH_PSI_PARTITION,),
        ),
    )

    assert projection.upload_partition_deltas[
        FEDMATCH_SIGMA_PARTITION
    ].peft_parameter_deltas["encoder_lora.weight"] == pytest.approx([0.0, 0.05])
    assert projection.client_partition_parameters[
        FEDMATCH_SIGMA_PARTITION
    ].peft_parameters["encoder_lora.weight"] == pytest.approx([0.10, 0.25])
    assert projection.upload_partition_deltas[
        FEDMATCH_PSI_PARTITION
    ].peft_parameter_deltas["encoder_lora.weight"] == pytest.approx([-0.10, 0.10])
    assert projection.client_partition_parameters[
        FEDMATCH_PSI_PARTITION
    ].peft_parameters["encoder_lora.weight"] == pytest.approx([0.0, 0.30])


def _build_physical_partitioned_model() -> ptm.PartitionedTrainableAdapterClassifier:
    return ptm.PartitionedTrainableAdapterClassifier(
        feature_extractor=TinyFrozenFeatureExtractor(),
        partitions=(
            _physical_partition_spec(FEDMATCH_SIGMA_PARTITION, weight_scale=1.0),
            _physical_partition_spec(FEDMATCH_PSI_PARTITION, weight_scale=0.5),
        ),
    )


def _physical_partition_spec(
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
