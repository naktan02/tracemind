"""PEFT encoder + classifier 공통 core 단위 검증."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from methods.adaptation.common.query_ssl_training_resume import (
    load_query_ssl_training_checkpoint,
    save_query_ssl_training_checkpoint,
)
from methods.adaptation.local_objective_regularizers.fedprox import (
    compute_fedprox_proximal_loss,
    snapshot_trainable_parameters,
)
from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.training import (
    pseudo_label_diagnostics as pld,
)
from methods.adaptation.peft_text_encoder.training.delta_extraction import (
    extract_peft_parameter_deltas,
)
from methods.adaptation.peft_text_encoder.training.loops import (
    evaluate_classifier,
    train_classifier,
    train_query_ssl_classifier,
)
from methods.adaptation.peft_text_encoder.training.modeling import (
    PeftTextEncoderWithLinearHead,
)
from methods.adaptation.peft_text_encoder.training.ssl_model_extensions import (
    build_peft_query_ssl_model_extensions,
)

resolve_fixed_pseudo_label_diagnostic_threshold = (
    pld.resolve_fixed_pseudo_label_diagnostic_threshold
)


class _TinyBackbone(nn.Module):
    """transformers output shape만 흉내 내는 작은 backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)
        self.embedding = nn.Embedding(8, 3)
        self.projection = nn.Linear(3, 3)

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        hidden = self.projection(self.embedding(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


def _build_loader() -> DataLoader[dict[str, torch.Tensor]]:
    rows = [
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor(0),
        },
        {
            "input_ids": torch.tensor([3, 4]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor(1),
        },
    ]
    return DataLoader(rows, batch_size=2)


def test_pseudo_label_diagnostic_threshold_marks_fixed_threshold_only() -> None:
    fixed = resolve_fixed_pseudo_label_diagnostic_threshold({"p_cutoff": 0.95})
    adaptive_without_fixed_cutoff = resolve_fixed_pseudo_label_diagnostic_threshold({})

    assert fixed.threshold == 0.95
    assert fixed.to_client_metrics() == {
        "query_ssl_diagnostic_uses_fixed_threshold": 1.0,
        "query_ssl_diagnostic_acceptance_threshold": 0.95,
    }
    assert adaptive_without_fixed_cutoff.threshold is None
    assert adaptive_without_fixed_cutoff.to_client_metrics() == {
        "query_ssl_diagnostic_uses_fixed_threshold": 0.0
    }


def test_peft_encoder_text_classifier_train_step_and_evaluation() -> None:
    torch.manual_seed(7)
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    loader = _build_loader()

    trained, history, selection_report = train_classifier(
        model=model,
        train_loader=loader,
        selection_loader=loader,
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=1,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
    )
    evaluation = evaluate_classifier(
        model=trained,
        dataloader=loader,
        categories=["anxiety", "normal"],
        device="cpu",
    )

    assert history[0]["epoch"] == 1
    assert selection_report["rows_total"] == 2
    assert evaluation["rows_total"] == 2
    assert set(evaluation["per_category"]) == {"anxiety", "normal"}


def test_peft_encoder_config_accepts_fedprox_mu() -> None:
    config = PeftEncoderTrainingBackendConfig.from_mapping({"proximal_mu": 0.01})

    assert config.proximal_mu == 0.01


class _CountingQuerySslAlgorithm:
    algorithm_name = "counting"

    def __init__(self) -> None:
        self.steps = 0
        self.num_train_iter = 0

    @property
    def uses_labeled_batches(self) -> bool:
        return True

    def configure_training(self, *, num_train_iter: int) -> None:
        self.num_train_iter = num_train_iter

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        assert train_loader_length > 0
        assert unlabeled_loader_length > 0

    def compute_step(self, *, model, labeled_batch, unlabeled_batch):
        del unlabeled_batch
        self.steps += 1
        assert labeled_batch is not None
        logits = model(
            input_ids=labeled_batch["input_ids"],
            attention_mask=labeled_batch["attention_mask"],
        )
        loss = logits.mean()
        return SimpleNamespace(
            total_loss=loss,
            loss_components={"dummy_loss": loss.detach()},
            metrics={"dummy_metric": loss.detach()},
        )


class _StatefulCountingQuerySslAlgorithm(_CountingQuerySslAlgorithm):
    algorithm_name = "stateful_counting"

    def __init__(self) -> None:
        super().__init__()
        self.dataset_configured = False
        self.events: list[str] = []

    def configure_dataset(
        self,
        *,
        num_classes: int,
        unlabeled_row_count: int,
    ) -> None:
        assert num_classes == 2
        assert unlabeled_row_count == 5
        self.dataset_configured = True
        self.events.append("dataset")

    def configure_labeled_dataset(self, *, labeled_row_count: int) -> None:
        assert labeled_row_count == 2
        self.events.append("labeled_dataset")

    def configure_training(self, *, num_train_iter: int) -> None:
        self.events.append("training")
        super().configure_training(num_train_iter=num_train_iter)

    def load_state(self, state):
        assert self.dataset_configured
        self.steps = int(state["steps"])
        self.events.append("load")

    def export_state(self):
        return {
            "schema_version": "query_ssl_algorithm_state.v1",
            "algorithm_name": self.algorithm_name,
            "stateful": True,
            "configured": True,
            "steps": self.steps,
        }


class _ContextAwareCountingQuerySslAlgorithm(_CountingQuerySslAlgorithm):
    algorithm_name = "context_aware_counting"

    def __init__(self) -> None:
        super().__init__()
        self.step_contexts = []

    def compute_step(self, *, model, labeled_batch, unlabeled_batch):
        raise AssertionError("context-aware algorithm should receive step context.")

    def compute_step_with_context(
        self,
        *,
        model,
        labeled_batch,
        unlabeled_batch,
        step_context,
    ):
        self.step_contexts.append(step_context)
        return super().compute_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
        )


class _InitialSelectionLossQuerySslAlgorithm(_CountingQuerySslAlgorithm):
    algorithm_name = "initial_selection_loss_counting"

    def __init__(self) -> None:
        super().__init__()
        self.initial_selection_loss: float | None = None

    @property
    def needs_initial_selection_loss(self) -> bool:
        return self.initial_selection_loss is None

    def configure_initial_selection_loss(self, *, selection_loss: float) -> None:
        self.initial_selection_loss = float(selection_loss)

    def compute_step(self, *, model, labeled_batch, unlabeled_batch):
        assert self.initial_selection_loss is not None
        return super().compute_step(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
        )


class _AuxiliaryProjectionQuerySslAlgorithm(_CountingQuerySslAlgorithm):
    algorithm_name = "auxiliary_projection"

    def __init__(self) -> None:
        super().__init__()
        self.projection: nn.Linear | None = None
        self.initial_projection_weight: torch.Tensor | None = None

    def build_auxiliary_modules(self, *, model):
        del model
        self.projection = nn.Linear(3, 1, bias=False)
        nn.init.constant_(self.projection.weight, 0.25)
        self.initial_projection_weight = self.projection.weight.detach().clone()
        return {"projection": self.projection}

    def compute_step(self, *, model, labeled_batch, unlabeled_batch):
        del unlabeled_batch
        self.steps += 1
        assert labeled_batch is not None
        assert self.projection is not None
        pooled = model.extract_pooled_features(
            input_ids=labeled_batch["input_ids"],
            attention_mask=labeled_batch["attention_mask"],
        )
        projected = self.projection(pooled)
        loss = projected.square().mean()
        return SimpleNamespace(
            total_loss=loss,
            loss_components={"projection_loss": loss.detach()},
            metrics={"projection_norm": self.projection.weight.detach().norm()},
        )


class _PostStepHookQuerySslAlgorithm(_CountingQuerySslAlgorithm):
    algorithm_name = "post_step_hook"

    def __init__(self) -> None:
        super().__init__()
        self.model_configured = False
        self.post_step_weight_norms: list[float] = []
        self.post_step_global_steps: list[int] = []

    def configure_model(self, *, model, device) -> None:
        del device
        self.model_configured = True
        assert isinstance(model, PeftTextEncoderWithLinearHead)

    def after_optimizer_step(self, *, model, step_context) -> None:
        assert self.model_configured
        self.post_step_weight_norms.append(
            float(model.classifier.weight.detach().norm())
        )
        self.post_step_global_steps.append(int(step_context.global_step))


def _build_unlabeled_loader() -> DataLoader[dict[str, torch.Tensor]]:
    rows = [
        {
            "weak_input_ids": torch.tensor([1, 2]),
            "weak_attention_mask": torch.tensor([1, 1]),
            "strong_input_ids": torch.tensor([2, 3]),
            "strong_attention_mask": torch.tensor([1, 1]),
        }
        for _ in range(5)
    ]
    return DataLoader(rows, batch_size=1)


def test_query_ssl_training_stops_at_max_train_steps() -> None:
    torch.manual_seed(7)
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    algorithm = _CountingQuerySslAlgorithm()

    _, history, _ = train_query_ssl_classifier(
        model=model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=5,
        max_train_steps=3,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=algorithm,
    )

    assert algorithm.num_train_iter == 3
    assert algorithm.steps == 3
    assert len(history) == 3


def test_query_ssl_training_loads_initial_state_after_dataset_config() -> None:
    torch.manual_seed(7)
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    algorithm = _StatefulCountingQuerySslAlgorithm()

    train_query_ssl_classifier(
        model=model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=1,
        max_train_steps=1,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=algorithm,
        initial_query_ssl_algorithm_state={
            "schema_version": "query_ssl_algorithm_state.v1",
            "algorithm_name": "stateful_counting",
            "stateful": True,
            "configured": True,
            "steps": 4,
        },
    )

    assert algorithm.events[:4] == ["dataset", "labeled_dataset", "training", "load"]
    assert algorithm.steps == 5


def test_query_ssl_training_passes_step_context_to_context_aware_algorithm() -> None:
    torch.manual_seed(7)
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    algorithm = _ContextAwareCountingQuerySslAlgorithm()

    train_query_ssl_classifier(
        model=model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=2,
        max_train_steps=2,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=algorithm,
    )

    assert [context.global_step for context in algorithm.step_contexts] == [1, 2]
    assert {context.total_train_steps for context in algorithm.step_contexts} == {2}
    assert {context.num_classes for context in algorithm.step_contexts} == {2}
    assert {context.device.type for context in algorithm.step_contexts} == {"cpu"}


def test_query_ssl_training_injects_initial_selection_loss_when_required() -> None:
    torch.manual_seed(7)
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    algorithm = _InitialSelectionLossQuerySslAlgorithm()

    train_query_ssl_classifier(
        model=model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=1,
        max_train_steps=1,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=algorithm,
    )

    assert algorithm.initial_selection_loss is not None
    assert algorithm.initial_selection_loss > 0
    assert algorithm.steps == 1


def test_query_ssl_training_runs_algorithm_post_optimizer_step_hook() -> None:
    torch.manual_seed(7)
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    algorithm = _PostStepHookQuerySslAlgorithm()

    train_query_ssl_classifier(
        model=model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=2,
        max_train_steps=2,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=algorithm,
    )

    assert algorithm.model_configured is True
    assert algorithm.post_step_global_steps == [1, 2]
    assert len(algorithm.post_step_weight_norms) == 2


def test_query_ssl_training_updates_algorithm_auxiliary_module() -> None:
    torch.manual_seed(7)
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    algorithm = _AuxiliaryProjectionQuerySslAlgorithm()

    train_query_ssl_classifier(
        model=model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=2,
        max_train_steps=2,
        learning_rate=0.05,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=algorithm,
    )

    assert algorithm.projection is not None
    assert algorithm.initial_projection_weight is not None
    assert not torch.allclose(
        algorithm.projection.weight.detach(),
        algorithm.initial_projection_weight,
    )


def test_query_ssl_auxiliary_modules_are_not_registered_on_peft_model() -> None:
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    extensions = build_peft_query_ssl_model_extensions(
        algorithm=_AuxiliaryProjectionQuerySslAlgorithm(),
        model=model,
        device="cpu",
    )

    assert list(extensions.auxiliary_modules) == ["projection"]
    assert "projection.weight" not in dict(model.named_parameters())
    assert "projection.weight" not in extract_peft_parameter_deltas(
        model=model,
        base_parameters={},
    )


def test_fedprox_proximal_loss_uses_round_start_snapshot() -> None:
    torch.manual_seed(7)
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    trainable_parameters = tuple(model.classifier.parameters())
    snapshot = snapshot_trainable_parameters(trainable_parameters)

    initial_loss = compute_fedprox_proximal_loss(
        trainable_parameters=trainable_parameters,
        reference_snapshot=snapshot,
    )
    with torch.no_grad():
        model.classifier.bias.add_(1.0)
    drifted_loss = compute_fedprox_proximal_loss(
        trainable_parameters=trainable_parameters,
        reference_snapshot=snapshot,
    )

    assert initial_loss.item() == 0.0
    assert drifted_loss.item() > 0.0


def test_query_ssl_training_records_fedprox_proximal_loss() -> None:
    torch.manual_seed(7)
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )

    _, history, _ = train_query_ssl_classifier(
        model=model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=3,
        max_train_steps=3,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=_CountingQuerySslAlgorithm(),
        proximal_mu=0.1,
    )

    proximal_losses = [
        float(record.get("train_fedprox_proximal_loss", 0.0)) for record in history
    ]
    assert proximal_losses[0] == 0.0
    assert max(proximal_losses[1:]) > 0.0


def test_query_ssl_training_resume_checkpoint_continues_remaining_steps(
    tmp_path,
) -> None:
    torch.manual_seed(7)
    checkpoint_dir = tmp_path / "checkpoints"
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    first_algorithm = _CountingQuerySslAlgorithm()

    _, first_history, _ = train_query_ssl_classifier(
        model=model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=2,
        max_train_steps=2,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=first_algorithm,
        resume_checkpoint_output_dir=checkpoint_dir,
        resume_checkpoint_every_epochs=1,
    )
    checkpoint_path = checkpoint_dir / "latest_training_checkpoint.pt"

    resumed_model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    resumed_algorithm = _CountingQuerySslAlgorithm()
    _, resumed_history, _ = train_query_ssl_classifier(
        model=resumed_model,
        train_loader=_build_loader(),
        unlabeled_loader=_build_unlabeled_loader(),
        selection_loader=_build_loader(),
        categories=["anxiety", "normal"],
        device="cpu",
        epochs=3,
        max_train_steps=3,
        learning_rate=0.01,
        classifier_learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every_steps=0,
        algorithm=resumed_algorithm,
        resume_checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
    assert first_algorithm.steps == 2
    assert len(first_history) == 2
    assert resumed_algorithm.steps == 1
    assert len(resumed_history) == 3


def test_query_ssl_training_checkpoint_roundtrips_auxiliary_module_state(
    tmp_path,
) -> None:
    model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    algorithm = _AuxiliaryProjectionQuerySslAlgorithm()
    extensions = build_peft_query_ssl_model_extensions(
        algorithm=algorithm,
        model=model,
        device="cpu",
    )
    assert algorithm.projection is not None
    with torch.no_grad():
        algorithm.projection.weight.fill_(1.75)
    optimizer = torch.optim.SGD(
        [
            *model.parameters(),
            *extensions.auxiliary_trainable_parameters,
        ],
        lr=0.1,
    )

    save_query_ssl_training_checkpoint(
        checkpoint_output_dir=tmp_path,
        algorithm=algorithm,
        model=model,
        optimizer=optimizer,
        completed_steps=1,
        total_train_steps=2,
        history=[],
        best_checkpoint_state={},
        categories=["anxiety", "normal"],
        auxiliary_modules=extensions.auxiliary_modules,
    )

    restored_model = PeftTextEncoderWithLinearHead(
        backbone=_TinyBackbone(),
        hidden_size=3,
        num_labels=2,
        classifier_dropout=0.0,
    )
    restored_algorithm = _AuxiliaryProjectionQuerySslAlgorithm()
    restored_extensions = build_peft_query_ssl_model_extensions(
        algorithm=restored_algorithm,
        model=restored_model,
        device="cpu",
    )
    restored_optimizer = torch.optim.SGD(
        [
            *restored_model.parameters(),
            *restored_extensions.auxiliary_trainable_parameters,
        ],
        lr=0.1,
    )

    resume_state = load_query_ssl_training_checkpoint(
        path=tmp_path / "latest_training_checkpoint.pt",
        model=restored_model,
        optimizer=restored_optimizer,
        algorithm=restored_algorithm,
        categories=["anxiety", "normal"],
        device="cpu",
        auxiliary_modules=restored_extensions.auxiliary_modules,
    )

    assert resume_state.completed_steps == 1
    assert restored_algorithm.projection is not None
    assert torch.allclose(
        restored_algorithm.projection.weight,
        torch.full_like(restored_algorithm.projection.weight, 1.75),
    )


def test_query_text_views_paths_do_not_keep_compatibility_shims() -> None:
    from pathlib import Path

    package_root = Path("methods/adaptation/query_text_views")

    assert not (package_root / "modeling.py").exists()
    assert not (package_root / "training.py").exists()
