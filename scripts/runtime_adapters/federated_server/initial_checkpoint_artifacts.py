"""FL bootstrap initial checkpoint를 server-owned tensor artifact로 승격한다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from main_server.src.services.federation.rounds import (
    initial_state_artifact_publication as initial_artifact_publication,
)
from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from methods.adaptation.peft_text_encoder.simulation_runtime.round_runtime import (
    FederatedPeftEncoderRuntimeConfig,
)
from methods.adaptation.peft_text_encoder.training.delta_extraction import (
    extract_peft_encoder_materialized_state,
)
from methods.adaptation.peft_text_encoder.training.modeling import (
    PeftTextEncoderWithLinearHead,
    require_transformer_stack,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.adaptation.peft_text_encoder.update.merged_tensor_artifact import (
    build_peft_adapter_state_tensor_artifact,
)
from methods.adaptation.text_encoder_classifier.classifier_head_tensor_artifact import (
    build_classifier_head_state_tensor_artifact,
)
from methods.adaptation.text_encoder_classifier.modeling import (
    load_classifier_head_state_if_configured,
    load_transformer_backbone,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedInitialCheckpointConfig,
    FederatedLocalTrainerRuntimeConfig,
    SimulationRunRequest,
)

CENTRAL_PEFT_CHECKPOINT_KIND = "central_peft_classifier_checkpoint"
INITIAL_CHECKPOINT_REF_ROOT = "server-aggregate://initial-checkpoint"

MaterializedCheckpointBuilder = Callable[
    [
        "ResolvedInitialCheckpointSource",
        FederatedPeftEncoderRuntimeConfig,
        FederatedLocalTrainerRuntimeConfig,
        tuple[str, ...],
    ],
    PeftEncoderMaterializedState,
]


@dataclass(frozen=True, slots=True)
class ResolvedInitialCheckpointSource:
    """FSSL bootstrap이 artifact로 승격할 중앙 checkpoint source."""

    manifest_path: Path | None
    adapter_dir: Path
    classifier_path: Path
    resolved_kind: str
    reference_id: str


def publish_initial_checkpoint_artifacts_for_request(
    *,
    request: SimulationRunRequest,
    labels: tuple[str, ...],
    materialized_checkpoint_builder: MaterializedCheckpointBuilder | None = None,
) -> None:
    """요청된 중앙 checkpoint를 FSSL 서버 artifact store에 publish한다."""

    source = _resolve_initial_checkpoint_source(request.initial_checkpoint_config)
    if source is None:
        return

    runtime_payload = _require_peft_runtime_payload(request)
    _require_empty_initial_refs(runtime_payload)
    state_builder = (
        materialized_checkpoint_builder
        or build_materialized_state_from_central_peft_checkpoint
    )
    materialized_state = state_builder(
        source,
        runtime_payload,
        request.local_trainer_runtime_config,
        labels,
    )
    artifact_store = AggregationArtifactStore(
        state_root=request.output_dir / "main_server" / "aggregation_artifacts"
    )
    peft_ref, head_ref = _save_initial_checkpoint_artifacts(
        artifact_store=artifact_store,
        source=source,
        state=materialized_state,
        labels=labels,
    )
    runtime_payload.peft_adapter_artifact_ref = peft_ref
    runtime_payload.classifier_head_artifact_ref = head_ref
    _record_published_checkpoint(
        config=request.initial_checkpoint_config,
        source=source,
        peft_ref=peft_ref,
        head_ref=head_ref,
    )


def build_materialized_state_from_central_peft_checkpoint(
    source: ResolvedInitialCheckpointSource,
    runtime_payload: FederatedPeftEncoderRuntimeConfig,
    runtime_config: FederatedLocalTrainerRuntimeConfig,
    labels: tuple[str, ...],
) -> PeftEncoderMaterializedState:
    """중앙 PEFT adapter/head checkpoint를 canonical FSSL state로 변환한다."""

    AutoModel, _AutoTokenizer, _LoraConfig, _TaskType, _get_peft_model, PeftModel = (
        require_transformer_stack()
    )
    backend_config = runtime_payload.training_backend_config
    backbone_base = load_transformer_backbone(
        model_cls=AutoModel,
        model_id=backend_config.backbone_model_id,
        revision=backend_config.backbone_revision,
        cache_dir=runtime_config.cache_dir,
        local_files_only=runtime_config.local_files_only,
        trust_remote_code=runtime_config.trust_remote_code,
    )
    backbone = PeftModel.from_pretrained(
        backbone_base,
        str(source.adapter_dir),
        is_trainable=True,
    )
    model = PeftTextEncoderWithLinearHead(
        backbone=backbone,
        hidden_size=int(backbone.config.hidden_size),
        num_labels=len(labels),
        classifier_dropout=runtime_config.classifier_dropout,
    ).to(runtime_config.device)
    load_classifier_head_state_if_configured(
        model=model,
        categories=list(labels),
        classifier_path=str(source.classifier_path),
    )
    try:
        return extract_peft_encoder_materialized_state(
            model=model,
            labels=labels,
        )
    finally:
        _release_checkpoint_model(model)


def _resolve_initial_checkpoint_source(
    config: FederatedInitialCheckpointConfig,
) -> ResolvedInitialCheckpointSource | None:
    manifest_path = _optional_path(config.manifest_path)
    adapter_dir = _optional_path(config.adapter_dir)
    classifier_path = _optional_path(config.classifier_path)
    manifest_payload: dict[str, object] | None = None
    source = "none"
    resolved_kind = "none"

    if manifest_path is not None:
        manifest_payload, manifest_path = _load_checkpoint_manifest(manifest_path)
        adapter_dir = adapter_dir or _optional_manifest_path(
            manifest_payload.get("adapter_dir"),
            base_dir=manifest_path.parent,
        )
        classifier_path = classifier_path or _optional_manifest_path(
            manifest_payload.get("classifier_path")
            or manifest_payload.get("model_path"),
            base_dir=manifest_path.parent,
        )
        source = "manifest"
        resolved_kind = _detect_manifest_kind(manifest_payload)
    elif adapter_dir is not None or classifier_path is not None:
        source = "config_group_paths"
        resolved_kind = "explicit_paths"

    if source == "none":
        if config.mode == "required":
            raise ValueError(
                "query_adaptation_initial_checkpoint is required for this FL SSL "
                "run. Provide manifest_path or adapter_dir/classifier_path."
            )
        return None

    if adapter_dir is None or classifier_path is None:
        raise ValueError(
            "FL SSL initial checkpoint requires both PEFT adapter_dir and "
            "classifier_path so the initial shared state can be built as canonical "
            "server tensor artifacts."
        )
    _ensure_existing_dir(adapter_dir, field_name="adapter_dir")
    _ensure_existing_file(classifier_path, field_name="classifier_path")
    if classifier_path.suffix != ".safetensors":
        raise ValueError(
            "FL SSL initial checkpoint requires classifier_head.safetensors. "
            f"Got: {classifier_path}"
        )

    reference_id = _reference_id(
        manifest_payload=manifest_payload,
        manifest_path=manifest_path,
        adapter_dir=adapter_dir,
    )
    config.source = source
    config.resolved_kind = resolved_kind
    config.reference_id = reference_id
    config.manifest_path = None if manifest_path is None else str(manifest_path)
    config.adapter_dir = str(adapter_dir)
    config.classifier_path = str(classifier_path)
    return ResolvedInitialCheckpointSource(
        manifest_path=manifest_path,
        adapter_dir=adapter_dir,
        classifier_path=classifier_path,
        resolved_kind=resolved_kind,
        reference_id=reference_id,
    )


def _require_peft_runtime_payload(
    request: SimulationRunRequest,
) -> FederatedPeftEncoderRuntimeConfig:
    payload = request.round_runtime_config.runtime_payload_for_update_family()
    if not isinstance(payload, FederatedPeftEncoderRuntimeConfig):
        raise ValueError(
            "FL SSL initial checkpoint currently requires the peft_text_encoder "
            "update family runtime payload."
        )
    return payload


def _require_empty_initial_refs(
    runtime_payload: FederatedPeftEncoderRuntimeConfig,
) -> None:
    if (
        runtime_payload.peft_adapter_artifact_ref is not None
        or runtime_payload.classifier_head_artifact_ref is not None
    ):
        raise ValueError(
            "Configure either query_adaptation_initial_checkpoint or explicit "
            "round_runtime initial artifact refs, not both."
        )


def _save_initial_checkpoint_artifacts(
    *,
    artifact_store: AggregationArtifactStore,
    source: ResolvedInitialCheckpointSource,
    state: PeftEncoderMaterializedState,
    labels: tuple[str, ...],
) -> tuple[str, str]:
    peft_tensors, peft_metadata = build_peft_adapter_state_tensor_artifact(
        peft_parameters=state.peft_parameters,
        applied_peft_parameter_deltas={},
    )
    head_tensors, head_metadata = build_classifier_head_state_tensor_artifact(
        classifier_head_weights=state.classifier_head_weights,
        classifier_head_biases=state.classifier_head_biases,
        label_schema=labels,
    )
    publication = initial_artifact_publication.InitialStateArtifactPublicationService(
        artifact_store=artifact_store
    ).publish_tensor_artifacts(
        initial_artifact_publication.InitialStateArtifactPublicationRequest(
            publication_id=source.reference_id,
            artifact_ref_prefix=INITIAL_CHECKPOINT_REF_ROOT,
            artifact_slots=(
                initial_artifact_publication.ServerTensorArtifactSlot(
                    artifact_name="peft-adapter-state",
                    tensors=peft_tensors,
                    metadata=peft_metadata,
                ),
                initial_artifact_publication.ServerTensorArtifactSlot(
                    artifact_name="classifier-head-state",
                    tensors=head_tensors,
                    metadata=head_metadata,
                ),
            ),
        )
    )
    return (
        publication.artifact_refs["peft-adapter-state"],
        publication.artifact_refs["classifier-head-state"],
    )


def _record_published_checkpoint(
    *,
    config: FederatedInitialCheckpointConfig,
    source: ResolvedInitialCheckpointSource,
    peft_ref: str,
    head_ref: str,
) -> None:
    config.source = "manifest" if source.manifest_path is not None else config.source
    config.resolved_kind = CENTRAL_PEFT_CHECKPOINT_KIND
    config.reference_id = source.reference_id
    config.peft_adapter_artifact_ref = peft_ref
    config.classifier_head_artifact_ref = head_ref


def _load_checkpoint_manifest(path: Path) -> tuple[dict[str, object], Path]:
    _ensure_existing_file(path, field_name="manifest_path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Initial checkpoint manifest must be a JSON object: {path}")
    nested_manifest_path = _optional_manifest_path(
        payload.get("manifest_path"),
        base_dir=path.parent,
    )
    if (
        nested_manifest_path is not None
        and "adapter_dir" not in payload
        and "classifier_path" not in payload
        and "model_path" not in payload
    ):
        return _load_checkpoint_manifest(nested_manifest_path)
    if isinstance(payload.get("manifest"), dict):
        return dict(payload["manifest"]), path
    return payload, path


def _detect_manifest_kind(payload: dict[str, object]) -> str:
    if str(payload.get("adapter_dir", "") or "").strip():
        return "central_peft_checkpoint_manifest"
    return "generic_checkpoint_manifest"


def _reference_id(
    *,
    manifest_payload: dict[str, object] | None,
    manifest_path: Path | None,
    adapter_dir: Path,
) -> str:
    if manifest_payload is not None:
        for key in ("trainer_version", "checkpoint_id", "run_id"):
            value = str(manifest_payload.get(key, "") or "").strip()
            if value:
                return value
    if manifest_path is not None:
        return manifest_path.parent.name or manifest_path.stem
    return adapter_dir.parent.name or adapter_dir.name


def _optional_path(value: object, *, base_dir: Path | None = None) -> Path | None:
    if value is None:
        return None
    raw_value = str(value).strip()
    if not raw_value:
        return None
    path = Path(raw_value)
    if not path.is_absolute() and base_dir is not None:
        return base_dir / path
    return path


def _optional_manifest_path(value: object, *, base_dir: Path) -> Path | None:
    path = _optional_path(value)
    if path is None:
        return None
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def _ensure_existing_dir(path: Path, *, field_name: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Resolved FL SSL initial checkpoint {field_name} does not exist: {path}"
        )


def _ensure_existing_file(path: Path, *, field_name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Resolved FL SSL initial checkpoint {field_name} does not exist: {path}"
        )


def _release_checkpoint_model(model: object) -> None:
    del model
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency guard
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
