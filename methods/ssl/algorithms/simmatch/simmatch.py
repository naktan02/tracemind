"""USB SimMatch core를 TraceMind reusable SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...base import (
    QUERY_SSL_ALGORITHM_STATE_DATASET_STATE,
    QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA,
    QUERY_SSL_ALGORITHM_STATE_FEATURE_QUEUE,
    QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
    QUERY_SSL_MODEL_OUTPUT_LOGITS,
    QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_AUXILIARY_TRAINABLE_MODULE,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
    QuerySslRuntimeRequirements,
    QuerySslStepContext,
    QuerySslStepResult,
    TextBatchClassifier,
)
from ...common import compute_prob
from ...hooks.consistency import CrossEntropyConsistencyLossHook
from ...hooks.distribution_alignment import QueueDistributionAlignmentHook
from ...hooks.masking import FixedThresholdMaskingHook
from ...model_capabilities import (
    FeatureReturningTextBatchClassifier,
    require_pooled_feature_classifier,
)
from ...projection import SslProjectionHead
from ...registry import register_query_ssl_algorithm
from ...state import (
    build_query_ssl_algorithm_state,
    require_matching_int_state_value,
    require_query_ssl_algorithm_state,
)
from ..usb_consistency import (
    USB_MULTIVIEW_REQUIRED_VIEWS,
    validate_usb_consistency_loaders,
)
from .memory_bank import SimMatchMemoryBank


class SimMatchAlgorithm:
    """SimMatch를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "simmatch"

    def __init__(
        self,
        *,
        T: float = 0.5,
        p_cutoff: float = 0.95,
        proj_size: int = 128,
        smoothing_alpha: float = 0.9,
        da_len: int = 256,
        in_loss_ratio: float = 1.0,
        lambda_u: float = 1.0,
        supervised_loss_weight: float = 1.0,
        ema_bank: float = 0.7,
    ) -> None:
        self.T = _require_positive_float(T, "T")
        self.p_cutoff = _require_probability_cutoff(p_cutoff, "p_cutoff")
        self.proj_size = _require_positive_int(proj_size, "proj_size")
        self.smoothing_alpha = _require_unit_interval(
            smoothing_alpha,
            "smoothing_alpha",
        )
        self.da_len = _require_positive_int(da_len, "da_len")
        self.in_loss_ratio = float(in_loss_ratio)
        self.lambda_u = float(lambda_u)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.ema_bank = _require_unit_interval(ema_bank, "ema_bank")
        self.num_classes: int | None = None
        self.labeled_row_count: int | None = None
        self.projection_head: nn.Module | None = None
        self.dist_align_hook: QueueDistributionAlignmentHook | None = None
        self.memory_bank: SimMatchMemoryBank | None = None
        self._pending_memory_bank_state: Mapping[str, Any] | None = None

    @property
    def uses_labeled_batches(self) -> bool:
        return True

    def configure_dataset(
        self,
        *,
        num_classes: int,
        unlabeled_row_count: int,
    ) -> None:
        del unlabeled_row_count
        self.num_classes = _require_positive_int(num_classes, "num_classes")
        self.dist_align_hook = QueueDistributionAlignmentHook(
            num_classes=self.num_classes,
            queue_length=self.da_len,
            p_target_type="uniform",
        )

    def configure_labeled_dataset(self, *, labeled_row_count: int) -> None:
        self.labeled_row_count = _require_positive_int(
            labeled_row_count,
            "labeled_row_count",
        )
        if (
            self.memory_bank is not None
            and self.memory_bank.bank_size != self.labeled_row_count
        ):
            raise RuntimeError(
                "SimMatch labeled dataset must be configured before memory bank "
                "creation."
            )

    def build_auxiliary_modules(
        self,
        *,
        model: TextBatchClassifier,
    ) -> dict[str, nn.Module]:
        classifier = require_pooled_feature_classifier(model)
        input_dim = int(getattr(classifier.classifier, "in_features", 0))
        if input_dim <= 0:
            raise ValueError("SimMatch requires classifier.in_features.")
        if self.projection_head is None:
            self.projection_head = SslProjectionHead(
                input_dim=input_dim,
                proj_size=self.proj_size,
            )
        self._ensure_memory_bank(device=next(self.projection_head.parameters()).device)
        return {"projection_head": self.projection_head}

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        validate_usb_consistency_loaders(
            algorithm_name="SimMatch",
            train_loader_length=train_loader_length,
            unlabeled_loader_length=unlabeled_loader_length,
            supervised_loss_weight=1.0,
        )

    def export_state(self) -> Mapping[str, Any]:
        memory_bank_state = (
            None if self.memory_bank is None else self.memory_bank.export_state()
        )
        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=(
                self.num_classes is not None and self.labeled_row_count is not None
            ),
            metadata={
                "num_classes": self.num_classes,
                "labeled_row_count": self.labeled_row_count,
                "proj_size": self.proj_size,
                "da_len": self.da_len,
                "dist_align": None
                if self.dist_align_hook is None
                else self.dist_align_hook.export_state(),
                "memory_bank": None
                if memory_bank_state is None
                else {
                    "feature_bank": memory_bank_state.feature_bank,
                    "labels_bank": memory_bank_state.labels_bank,
                },
            },
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        effective_state = require_query_ssl_algorithm_state(
            state=state,
            algorithm_name=self.algorithm_name,
        )
        if self.num_classes is not None:
            require_matching_int_state_value(
                state=effective_state,
                field_name="num_classes",
                expected=self.num_classes,
                algorithm_name=self.algorithm_name,
            )
        if self.labeled_row_count is not None:
            require_matching_int_state_value(
                state=effective_state,
                field_name="labeled_row_count",
                expected=self.labeled_row_count,
                algorithm_name=self.algorithm_name,
            )
        require_matching_int_state_value(
            state=effective_state,
            field_name="proj_size",
            expected=self.proj_size,
            algorithm_name=self.algorithm_name,
        )
        require_matching_int_state_value(
            state=effective_state,
            field_name="da_len",
            expected=self.da_len,
            algorithm_name=self.algorithm_name,
        )
        dist_align_state = effective_state.get("dist_align")
        if dist_align_state is not None:
            if self.dist_align_hook is None:
                raise RuntimeError("SimMatch dist_align_hook is not configured.")
            if not isinstance(dist_align_state, Mapping):
                raise ValueError("SimMatch dist_align state must be a mapping.")
            self.dist_align_hook.load_state(
                dist_align_state,
                device=torch.device("cpu"),
            )
        memory_bank_state = effective_state.get("memory_bank")
        if memory_bank_state is not None:
            if not isinstance(memory_bank_state, Mapping):
                raise ValueError("SimMatch memory_bank state must be a mapping.")
            if self.memory_bank is None:
                self._pending_memory_bank_state = memory_bank_state
            else:
                self.memory_bank.load_state(memory_bank_state)

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Tensor],
    ) -> QuerySslStepResult:
        return self.compute_step_with_context(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            step_context=None,
        )

    def compute_step_with_context(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Tensor],
        step_context: QuerySslStepContext | None,
    ) -> QuerySslStepResult:
        if self.dist_align_hook is None:
            raise RuntimeError("SimMatch must be configured with dataset metadata.")
        if self.projection_head is None:
            raise RuntimeError("SimMatch projection head was not initialized.")
        memory_bank = self._ensure_memory_bank(
            device=next(self.projection_head.parameters()).device
        )
        return compute_simmatch_step(
            model=require_pooled_feature_classifier(model),
            projection_head=self.projection_head,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            dist_align_hook=self.dist_align_hook,
            memory_bank=memory_bank,
            temperature=self.T,
            p_cutoff=self.p_cutoff,
            smoothing_alpha=self.smoothing_alpha,
            ema_bank=self.ema_bank,
            apply_similarity_smoothing=_should_apply_similarity_smoothing(
                step_context,
            ),
            lambda_u=self.lambda_u,
            lambda_in=self.in_loss_ratio,
            supervised_loss_weight=self.supervised_loss_weight,
        )

    def _ensure_memory_bank(self, *, device: torch.device) -> SimMatchMemoryBank:
        if self.labeled_row_count is None:
            raise RuntimeError(
                "SimMatch requires configure_labeled_dataset before training."
            )
        if self.memory_bank is None:
            self.memory_bank = SimMatchMemoryBank(
                bank_size=self.labeled_row_count,
                feature_dim=self.proj_size,
                device=device,
            )
            if self._pending_memory_bank_state is not None:
                self.memory_bank.load_state(self._pending_memory_bank_state)
                self._pending_memory_bank_state = None
        self.memory_bank.to(device)
        return self.memory_bank


def compute_simmatch_step(
    *,
    model: FeatureReturningTextBatchClassifier,
    projection_head: nn.Module,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Tensor],
    dist_align_hook: QueueDistributionAlignmentHook,
    memory_bank: SimMatchMemoryBank,
    temperature: float,
    p_cutoff: float,
    smoothing_alpha: float,
    ema_bank: float,
    apply_similarity_smoothing: bool,
    lambda_u: float = 1.0,
    lambda_in: float = 1.0,
    supervised_loss_weight: float = 1.0,
) -> QuerySslStepResult:
    """USB `semilearn/algorithms/simmatch/simmatch.py::train_step` 핵심."""

    if labeled_batch is None:
        raise ValueError("SimMatch requires a labeled_batch.")
    labels = labeled_batch["labels"].long()
    labeled_indices = _require_labeled_row_indices(labeled_batch).to(labels.device)
    logits_x_lb, feats_x_lb = _forward_logits_and_projected_features(
        model=model,
        projection_head=projection_head,
        input_ids=labeled_batch["input_ids"],
        attention_mask=labeled_batch["attention_mask"],
    )
    logits_x_ulb_w, feats_x_ulb_w = _forward_logits_and_projected_features(
        model=model,
        projection_head=projection_head,
        input_ids=unlabeled_batch["weak_input_ids"],
        attention_mask=unlabeled_batch["weak_attention_mask"],
    )
    logits_x_ulb_s, feats_x_ulb_s = _forward_logits_and_projected_features(
        model=model,
        projection_head=projection_head,
        input_ids=unlabeled_batch["strong_input_ids"],
        attention_mask=unlabeled_batch["strong_attention_mask"],
    )

    sup_loss = F.cross_entropy(logits_x_lb, labels, reduction="mean")
    probs_x_ulb_w = compute_prob(logits_x_ulb_w.detach())
    probs_x_ulb_w = dist_align_hook.dist_align(
        probs_x_ulb=probs_x_ulb_w.detach(),
    )
    bank_features = memory_bank.feature_bank.detach().clone()
    labels_bank = memory_bank.labels_bank.detach().long()
    first_epoch = not apply_similarity_smoothing

    with torch.no_grad():
        teacher_logits = feats_x_ulb_w.detach() @ bank_features
        teacher_prob_orig = F.softmax(teacher_logits / float(temperature), dim=1)
        factor = probs_x_ulb_w.gather(
            1,
            labels_bank.expand([probs_x_ulb_w.shape[0], -1]),
        )
        teacher_prob = teacher_prob_orig * factor
        teacher_prob = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp(
            min=1e-12,
        )
        if apply_similarity_smoothing and smoothing_alpha < 1:
            aggregated_prob = torch.zeros(
                (teacher_prob_orig.shape[0], logits_x_ulb_w.shape[1]),
                device=teacher_prob_orig.device,
                dtype=teacher_prob_orig.dtype,
            )
            aggregated_prob = aggregated_prob.scatter_add(
                1,
                labels_bank.expand([teacher_prob_orig.shape[0], -1]),
                teacher_prob_orig,
            )
            probs_for_consistency = probs_x_ulb_w * float(
                smoothing_alpha
            ) + aggregated_prob * (1.0 - float(smoothing_alpha))
        else:
            probs_for_consistency = probs_x_ulb_w

    student_logits = feats_x_ulb_s @ bank_features
    student_prob = F.softmax(student_logits / float(temperature), dim=1)
    in_loss = (
        (-teacher_prob.detach() * torch.log(student_prob + 1e-12)).sum(dim=1).mean()
    )
    if first_epoch:
        in_loss = in_loss * 0.0
        probs_for_consistency = probs_x_ulb_w

    mask = FixedThresholdMaskingHook().build_mask(
        probs_x_ulb_w=probs_for_consistency,
        p_cutoff=p_cutoff,
    )
    unsup_loss = CrossEntropyConsistencyLossHook().compute_loss(
        logits=logits_x_ulb_s,
        targets=probs_for_consistency.detach(),
        mask=mask,
    )
    total_loss = (
        float(supervised_loss_weight) * sup_loss
        + float(lambda_u) * unsup_loss
        + float(lambda_in) * in_loss
    )
    memory_bank.update(
        features=feats_x_lb.detach(),
        labels=labels.detach(),
        indices=labeled_indices.detach(),
        ema_bank=ema_bank,
    )
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "in_loss": in_loss,
        },
        metrics={
            "util_ratio": mask.float().mean(),
            "memory_bank_size": logits_x_lb.new_tensor(float(memory_bank.bank_size)),
            "similarity_smoothing_applied": logits_x_lb.new_tensor(
                float(apply_similarity_smoothing)
            ),
        },
        debug_tensors={
            "mask": mask,
            "teacher_prob": teacher_prob,
            "probs_x_ulb_w": probs_for_consistency,
        },
    )


def _forward_logits_and_projected_features(
    *,
    model: FeatureReturningTextBatchClassifier,
    projection_head: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    pooled_features = model.extract_pooled_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    projected_features = projection_head(pooled_features)
    if projected_features.ndim != 2:
        raise ValueError("SimMatch projection head must return [batch, proj_size].")
    return logits, F.normalize(projected_features, dim=1)


def _require_labeled_row_indices(labeled_batch: Mapping[str, Any]) -> Tensor:
    row_indices = labeled_batch.get("row_indices")
    if not isinstance(row_indices, Tensor):
        raise ValueError("SimMatch requires labeled_batch['row_indices'].")
    return row_indices.long()


def _should_apply_similarity_smoothing(
    step_context: QuerySslStepContext | None,
) -> bool:
    return step_context is None or step_context.epoch_index > 1


def _require_positive_int(value: int, field_name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return normalized


def _require_positive_float(value: float, field_name: str) -> float:
    normalized = float(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return normalized


def _require_probability_cutoff(value: float, field_name: str) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return normalized


def _require_unit_interval(value: float, field_name: str) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return normalized


@register_query_ssl_algorithm(
    "simmatch",
    display_name="SimMatch",
    required_views=USB_MULTIVIEW_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
    runtime_requirements=QuerySslRuntimeRequirements(
        batch_surface=QUERY_SSL_BATCH_SURFACE_WEAK_STRONG,
        model_outputs=frozenset(
            {
                QUERY_SSL_MODEL_OUTPUT_LOGITS,
                QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
            }
        ),
        algorithm_state_surface=frozenset(
            {
                QUERY_SSL_ALGORITHM_STATE_DATASET_STATE,
                QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA,
                QUERY_SSL_ALGORITHM_STATE_FEATURE_QUEUE,
            }
        ),
        optimizer_lifecycle=frozenset(
            {
                QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
                QUERY_SSL_OPTIMIZER_LIFECYCLE_AUXILIARY_TRAINABLE_MODULE,
            }
        ),
        step_context_required=True,
    ),
)
def build_simmatch_algorithm(parameters: Mapping[str, Any]) -> SimMatchAlgorithm:
    """Hydra method parameter mapping으로 SimMatch algorithm을 만든다."""

    return SimMatchAlgorithm(
        T=float(parameters.get("T", 0.5)),
        p_cutoff=float(parameters.get("p_cutoff", 0.95)),
        proj_size=int(parameters.get("proj_size", 128)),
        smoothing_alpha=float(parameters.get("smoothing_alpha", 0.9)),
        da_len=int(parameters.get("da_len", 256)),
        in_loss_ratio=float(parameters.get("in_loss_ratio", 1.0)),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
        ema_bank=float(parameters.get("ema_bank", 0.7)),
    )
