"""PEFT-classifier supervised server seed step primitive."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from methods.adaptation.query_classifier_adaptation.data import build_dataloader
from methods.adaptation.text_classifier.peft_encoder.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.text_classifier.peft_encoder.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.text_classifier.peft_encoder.training.delta_extraction import (
    extract_lora_classifier_parameter_deltas,
    load_lora_classifier_base_parameters_into_model,
)
from methods.adaptation.text_classifier.peft_encoder.training.loops import (
    set_seed,
    train_classifier,
)
from methods.adaptation.text_classifier.peft_encoder.training.modeling import (
    build_peft_encoder_text_classifier_from_config,
)
from methods.adaptation.text_classifier.peft_encoder.update.materialization import (
    LoraClassifierMaterializedState,
)
from methods.common.runtime_resources import RuntimeResourceCache
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

LoraClassifierTrainerRuntimeConfig = qssl_training.LoraClassifierTrainerRuntimeConfig


@dataclass(frozen=True, slots=True)
class LoraClassifierSupervisedSeedStepResult:
    """server bootstrap rows로 학습한 LoRA/classifier delta와 실행 metric."""

    lora_parameter_deltas: Mapping[str, Sequence[float]]
    classifier_head_weight_deltas: Mapping[str, Sequence[float]]
    classifier_head_bias_deltas: Mapping[str, float]
    metrics: dict[str, float]


PeftEncoderSupervisedSeedStepResult = LoraClassifierSupervisedSeedStepResult


def run_lora_classifier_supervised_seed_step_core(
    *,
    labels: Sequence[str],
    base_parameters: LoraClassifierMaterializedState,
    bootstrap_rows: Sequence[LabeledQueryRow],
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_clip_norm: float | None,
) -> LoraClassifierSupervisedSeedStepResult:
    """server-owned labeled seed rows로 LoRA classifier delta를 계산한다."""

    effective_labels = tuple(str(label) for label in labels)
    if not effective_labels:
        raise ValueError("supervised_seed_step label schema must not be empty.")
    if not bootstrap_rows:
        raise ValueError("supervised_seed_step requires server bootstrap_rows.")
    if epochs <= 0:
        raise ValueError("supervised_seed_step server epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("supervised_seed_step server batch size must be positive.")

    set_seed(int(seed))
    model, tokenizer = build_peft_encoder_text_classifier_from_config(
        labels=list(effective_labels),
        lora_config=lora_config,
        runtime_config=trainer_runtime_config,
        runtime_resource_cache=runtime_resource_cache,
    )
    load_lora_classifier_base_parameters_into_model(
        model=model,
        labels=effective_labels,
        base_parameters=base_parameters,
        device=trainer_runtime_config.device,
    )
    label_to_index = {label: index for index, label in enumerate(effective_labels)}
    train_loader = build_dataloader(
        rows=bootstrap_rows,
        label_to_index=label_to_index,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=lora_config.max_length,
        task_prefix=lora_config.task_prefix,
        shuffle=True,
    )
    train_classifier(
        model=model,
        train_loader=train_loader,
        selection_loader=train_loader,
        categories=list(effective_labels),
        device=trainer_runtime_config.device,
        epochs=epochs,
        learning_rate=learning_rate,
        classifier_learning_rate=learning_rate,
        weight_decay=0.0,
        max_grad_norm=0.0 if gradient_clip_norm is None else float(gradient_clip_norm),
        log_every_steps=0,
    )
    lora_deltas, head_weight_deltas, head_bias_deltas = (
        extract_lora_classifier_parameter_deltas(
            model=model,
            base_parameters=base_parameters,
            labels=effective_labels,
        )
    )
    return LoraClassifierSupervisedSeedStepResult(
        lora_parameter_deltas=lora_deltas,
        classifier_head_weight_deltas=head_weight_deltas,
        classifier_head_bias_deltas=head_bias_deltas,
        metrics={
            "server_step_supervised_seed": 1.0,
            "server_step_labeled_count": float(len(bootstrap_rows)),
            "server_step_epochs": float(epochs),
            "server_step_batch_size": float(batch_size),
        },
    )


run_peft_encoder_supervised_seed_step_core = (
    run_lora_classifier_supervised_seed_step_core
)
