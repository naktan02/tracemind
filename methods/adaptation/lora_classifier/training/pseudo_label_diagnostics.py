"""LoRA-classifier pseudo-label 품질 진단 helper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch

from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.lora_classifier.training.loops import (
    move_tensor_batch_to_device,
)
from methods.adaptation.lora_classifier.training.modeling import LoraTextClassifier
from methods.adaptation.query_classifier_adaptation.data import build_weak_dataloader
from methods.adaptation.query_classifier_adaptation.tokenization import (
    TextTokenizationCache,
)
from methods.evaluation.pseudo_label_quality import (
    PseudoLabelCandidateRecord,
    PseudoLabelQualitySummary,
    build_pseudo_label_quality_summary,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


class LoraClassifierDiagnosticsRuntimeConfig(Protocol):
    """pseudo-label diagnostics가 필요한 runtime surface."""

    device: str


def build_final_snapshot_pseudo_label_quality(
    *,
    model: LoraTextClassifier,
    tokenizer: Any,
    rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    lora_config: LoraClassifierTrainingBackendConfig,
    acceptance_threshold: float | None,
    trainer_runtime_config: LoraClassifierDiagnosticsRuntimeConfig,
    unlabeled_batch_size: int,
    tokenization_cache: TextTokenizationCache | None,
    tokenization_cache_namespace: str,
) -> PseudoLabelQualitySummary:
    """round 종료 시점 local classifier snapshot의 pseudo-label 품질을 계산한다."""

    effective_rows = list(rows)
    if not effective_rows:
        return PseudoLabelQualitySummary.empty()

    loader = build_weak_dataloader(
        rows=effective_rows,
        tokenizer=tokenizer,
        batch_size=int(unlabeled_batch_size),
        max_length=lora_config.max_length,
        task_prefix=lora_config.task_prefix,
        shuffle=False,
        tokenization_cache=tokenization_cache,
        tokenization_cache_namespace=tokenization_cache_namespace,
    )
    candidates: list[PseudoLabelCandidateRecord] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            device_batch = move_tensor_batch_to_device(
                batch=batch,
                device=trainer_runtime_config.device,
            )
            probabilities = torch.softmax(
                model(
                    input_ids=device_batch["weak_input_ids"],
                    attention_mask=device_batch["weak_attention_mask"],
                ),
                dim=-1,
            )
            top_k = min(2, probabilities.shape[-1])
            top_values, top_indices = torch.topk(probabilities, k=top_k, dim=-1)
            query_ids = [str(query_id) for query_id in batch["query_ids"]]
            for row_index, query_id in enumerate(query_ids):
                top1_index = int(top_indices[row_index, 0].detach().cpu().item())
                top1_score = float(top_values[row_index, 0].detach().cpu().item())
                top2_score = (
                    float(top_values[row_index, 1].detach().cpu().item())
                    if top_k > 1
                    else 0.0
                )
                candidates.append(
                    PseudoLabelCandidateRecord(
                        source_event_ref=query_id,
                        label=str(labels[top1_index]),
                        confidence=top1_score,
                        margin=top1_score - top2_score,
                        accepted=(
                            acceptance_threshold is not None
                            and top1_score >= acceptance_threshold
                        ),
                    )
                )

    return build_pseudo_label_quality_summary(
        candidates=tuple(candidates),
        rows_with_simulation_labels=effective_rows,
    )


def tokenization_cache_namespace(
    lora_config: LoraClassifierTrainingBackendConfig,
) -> str:
    """LoRA classifier tokenizer 설정을 cache namespace로 정규화한다."""

    return (
        f"tokenizer={lora_config.tokenizer_model_id}"
        f"|revision={lora_config.tokenizer_revision}"
    )
