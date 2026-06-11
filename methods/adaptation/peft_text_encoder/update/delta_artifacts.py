"""PEFT-encoder classifier delta artifact materialization rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from methods.adaptation.peft_text_encoder.config import (
    PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL,
    PEFT_ENCODER_DELTA_FORMAT_INLINE,
    PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierDelta,
)
from shared.src.contracts.training_contracts import TrainingTask

from ..training.query_ssl_federated_update import (
    QuerySslPeftEncoderDeltaMaterialization,
)
from .json_delta_artifact import (
    build_classifier_head_delta_json_artifact_payload,
    build_partitioned_delta_json_artifact_payload,
    build_peft_adapter_delta_json_artifact_payload,
)
from .merged_tensor_artifact import (
    build_classifier_head_delta_tensor_artifact,
    build_peft_adapter_delta_tensor_artifact,
)
from .partitioned_delta import PeftEncoderPartitionDelta
from .partitioned_tensor_artifact import build_partitioned_delta_tensor_artifact


class PeftEncoderDeltaArtifactStore(Protocol):
    """PEFT encoder delta materializer가 필요한 runtime artifact store surface."""

    def ref_for_agent_artifact(
        self,
        *,
        artifact_ref_prefix: str,
        training_task: TrainingTask,
        client_id: str,
        update_id: str,
        artifact_name: str,
    ) -> str:
        """agent-local artifact ref를 만든다."""

    def save_agent_json_artifact(
        self,
        *,
        artifact_ref: str,
        payload: Mapping[str, object],
    ) -> None:
        """agent-local JSON artifact를 저장한다."""

    def ref_for_server_client_update_artifact(
        self,
        *,
        training_task: TrainingTask,
        client_id: str,
        update_id: str,
        artifact_name: str,
    ) -> str:
        """server-owned client update artifact ref를 만든다."""

    def save_server_safetensors_artifact_ref(
        self,
        *,
        artifact_ref: str,
        tensors: Mapping[str, object],
        metadata: Mapping[str, str],
    ) -> None:
        """server-owned safetensors artifact를 저장한다."""

    def is_agent_local_ref(self, artifact_ref: str | None) -> bool:
        """runtime-local ref 여부를 판정한다."""

    def upload_agent_local_json_artifact(self, *, agent_local_ref: str) -> str:
        """agent-local JSON artifact를 server-owned ref로 복사한다."""

    def server_artifact_refs_byte_count(
        self,
        *,
        artifact_refs: Sequence[str | None],
    ) -> int:
        """server-owned artifact ref들의 저장 크기를 합산한다."""


@dataclass(frozen=True, slots=True)
class PeftEncoderDeltaMaterializer:
    """PEFT encoder/classifier delta를 runtime store에 맞게 materialize한다."""

    artifact_store: PeftEncoderDeltaArtifactStore

    def prepare(
        self,
        *,
        update_id: str,
        training_task: TrainingTask,
        client_id: str,
        delta_format: str,
        artifact_ref_prefix: str | None = None,
        peft_parameter_deltas: Mapping[str, Sequence[float]],
        classifier_head_weight_deltas: Mapping[str, Sequence[float]],
        classifier_head_bias_deltas: Mapping[str, float],
        partitioned_deltas: Mapping[str, PeftEncoderPartitionDelta] | None = None,
        materialize_primary_deltas: bool = True,
    ) -> QuerySslPeftEncoderDeltaMaterialization:
        """delta_format에 맞게 PEFT encoder/classifier delta artifact ref를 준비한다."""

        normalized_delta_format = str(delta_format).strip()
        if normalized_delta_format == PEFT_ENCODER_DELTA_FORMAT_INLINE:
            return QuerySslPeftEncoderDeltaMaterialization(
                delta_format=normalized_delta_format,
                peft_adapter_delta_artifact_ref=None,
                classifier_head_delta_artifact_ref=None,
                include_inline_deltas=True,
            )
        if normalized_delta_format == PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL:
            if artifact_ref_prefix is None or not artifact_ref_prefix.strip():
                raise ValueError(
                    "delta_format=agent_local requires artifact_ref_prefix."
                )
            if not materialize_primary_deltas and partitioned_deltas is None:
                raise ValueError(
                    "Skipping primary PEFT/head artifacts requires partitioned_deltas."
                )
            return self._prepare_agent_local_delta_materialization(
                update_id=update_id,
                training_task=training_task,
                client_id=client_id,
                artifact_ref_prefix=artifact_ref_prefix,
                peft_parameter_deltas=peft_parameter_deltas,
                classifier_head_weight_deltas=classifier_head_weight_deltas,
                classifier_head_bias_deltas=classifier_head_bias_deltas,
                partitioned_deltas=partitioned_deltas,
                materialize_primary_deltas=materialize_primary_deltas,
            )
        if normalized_delta_format != PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED:
            raise ValueError(
                "Unsupported Query SSL PEFT encoder delta_format: "
                f"{normalized_delta_format!r}."
            )
        if not materialize_primary_deltas and partitioned_deltas is None:
            raise ValueError(
                "Skipping primary PEFT/head artifacts requires partitioned_deltas."
            )
        return self._prepare_server_uploaded_delta_materialization(
            update_id=update_id,
            training_task=training_task,
            client_id=client_id,
            peft_parameter_deltas=peft_parameter_deltas,
            classifier_head_weight_deltas=classifier_head_weight_deltas,
            classifier_head_bias_deltas=classifier_head_bias_deltas,
            partitioned_deltas=partitioned_deltas,
            materialize_primary_deltas=materialize_primary_deltas,
        )

    def _prepare_agent_local_delta_materialization(
        self,
        *,
        update_id: str,
        training_task: TrainingTask,
        client_id: str,
        artifact_ref_prefix: str,
        peft_parameter_deltas: Mapping[str, Sequence[float]],
        classifier_head_weight_deltas: Mapping[str, Sequence[float]],
        classifier_head_bias_deltas: Mapping[str, float],
        partitioned_deltas: Mapping[str, PeftEncoderPartitionDelta] | None,
        materialize_primary_deltas: bool,
    ) -> QuerySslPeftEncoderDeltaMaterialization:
        peft_adapter_delta_ref = None
        head_delta_ref = None
        if materialize_primary_deltas:
            peft_adapter_delta_ref = self.artifact_store.ref_for_agent_artifact(
                artifact_ref_prefix=artifact_ref_prefix,
                training_task=training_task,
                client_id=client_id,
                update_id=update_id,
                artifact_name="peft_adapter_delta",
            )
            head_delta_ref = self.artifact_store.ref_for_agent_artifact(
                artifact_ref_prefix=artifact_ref_prefix,
                training_task=training_task,
                client_id=client_id,
                update_id=update_id,
                artifact_name="classifier_head_delta",
            )
            self.artifact_store.save_agent_json_artifact(
                artifact_ref=peft_adapter_delta_ref,
                payload=build_peft_adapter_delta_json_artifact_payload(
                    update_id=update_id,
                    training_task=training_task,
                    client_id=client_id,
                    peft_parameter_deltas=peft_parameter_deltas,
                ),
            )
            self.artifact_store.save_agent_json_artifact(
                artifact_ref=head_delta_ref,
                payload=build_classifier_head_delta_json_artifact_payload(
                    update_id=update_id,
                    training_task=training_task,
                    client_id=client_id,
                    classifier_head_weight_deltas=classifier_head_weight_deltas,
                    classifier_head_bias_deltas=classifier_head_bias_deltas,
                ),
            )
        partitioned_delta_ref = None
        if partitioned_deltas is not None:
            partitioned_delta_ref = self.artifact_store.ref_for_agent_artifact(
                artifact_ref_prefix=artifact_ref_prefix,
                training_task=training_task,
                client_id=client_id,
                update_id=update_id,
                artifact_name="partitioned_delta",
            )
            self.artifact_store.save_agent_json_artifact(
                artifact_ref=partitioned_delta_ref,
                payload=build_partitioned_delta_json_artifact_payload(
                    update_id=update_id,
                    training_task=training_task,
                    client_id=client_id,
                    partitioned_deltas=partitioned_deltas,
                ),
            )
        return QuerySslPeftEncoderDeltaMaterialization(
            delta_format=PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL,
            peft_adapter_delta_artifact_ref=peft_adapter_delta_ref,
            classifier_head_delta_artifact_ref=head_delta_ref,
            include_inline_deltas=False,
            partitioned_deltas_artifact_ref=partitioned_delta_ref,
        )

    def _prepare_server_uploaded_delta_materialization(
        self,
        *,
        update_id: str,
        training_task: TrainingTask,
        client_id: str,
        peft_parameter_deltas: Mapping[str, Sequence[float]],
        classifier_head_weight_deltas: Mapping[str, Sequence[float]],
        classifier_head_bias_deltas: Mapping[str, float],
        partitioned_deltas: Mapping[str, PeftEncoderPartitionDelta] | None,
        materialize_primary_deltas: bool,
    ) -> QuerySslPeftEncoderDeltaMaterialization:
        peft_adapter_delta_ref = (
            self.artifact_store.ref_for_server_client_update_artifact(
                training_task=training_task,
                client_id=client_id,
                update_id=update_id,
                artifact_name="peft_adapter_delta",
            )
            if materialize_primary_deltas
            else None
        )
        head_delta_ref = (
            self.artifact_store.ref_for_server_client_update_artifact(
                training_task=training_task,
                client_id=client_id,
                update_id=update_id,
                artifact_name="classifier_head_delta",
            )
            if materialize_primary_deltas
            else None
        )
        partitioned_delta_ref = (
            None
            if partitioned_deltas is None
            else self.artifact_store.ref_for_server_client_update_artifact(
                training_task=training_task,
                client_id=client_id,
                update_id=update_id,
                artifact_name="partitioned_delta",
            )
        )
        if peft_adapter_delta_ref is not None:
            tensors, metadata = build_peft_adapter_delta_tensor_artifact(
                peft_parameter_deltas
            )
            self.artifact_store.save_server_safetensors_artifact_ref(
                artifact_ref=peft_adapter_delta_ref,
                tensors=tensors,
                metadata=metadata,
            )
        if head_delta_ref is not None:
            tensors, metadata = build_classifier_head_delta_tensor_artifact(
                classifier_head_weight_deltas=classifier_head_weight_deltas,
                classifier_head_bias_deltas=classifier_head_bias_deltas,
            )
            self.artifact_store.save_server_safetensors_artifact_ref(
                artifact_ref=head_delta_ref,
                tensors=tensors,
                metadata=metadata,
            )
        if partitioned_delta_ref is not None:
            if partitioned_deltas is None:
                raise AssertionError("partitioned_deltas must exist before save.")
            tensors, metadata = build_partitioned_delta_tensor_artifact(
                partitioned_deltas
            )
            self.artifact_store.save_server_safetensors_artifact_ref(
                artifact_ref=partitioned_delta_ref,
                tensors=tensors,
                metadata=metadata,
            )
        return QuerySslPeftEncoderDeltaMaterialization(
            delta_format=PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED,
            peft_adapter_delta_artifact_ref=peft_adapter_delta_ref,
            classifier_head_delta_artifact_ref=head_delta_ref,
            include_inline_deltas=False,
            partitioned_deltas_artifact_ref=partitioned_delta_ref,
        )


def upload_agent_local_peft_encoder_update(
    *,
    artifact_store: PeftEncoderDeltaArtifactStore,
    update_payload: PeftClassifierDelta,
) -> PeftClassifierDelta:
    """agent-local delta artifact ref를 server-owned ref로 materialize한다."""

    update_fields: dict[str, object] = {}
    adapter_delta_field = _adapter_delta_artifact_ref_field(update_payload)
    adapter_delta_ref = _adapter_delta_artifact_ref(update_payload)
    if artifact_store.is_agent_local_ref(adapter_delta_ref):
        if adapter_delta_ref is None:
            raise AssertionError(f"{adapter_delta_field} unexpectedly missing.")
        update_fields[adapter_delta_field] = (
            artifact_store.upload_agent_local_json_artifact(
                agent_local_ref=adapter_delta_ref,
            )
        )
    if artifact_store.is_agent_local_ref(
        update_payload.classifier_head_delta_artifact_ref
    ):
        if update_payload.classifier_head_delta_artifact_ref is None:
            raise AssertionError(
                "classifier_head_delta_artifact_ref unexpectedly missing."
            )
        update_fields["classifier_head_delta_artifact_ref"] = (
            artifact_store.upload_agent_local_json_artifact(
                agent_local_ref=update_payload.classifier_head_delta_artifact_ref,
            )
        )
    if artifact_store.is_agent_local_ref(
        update_payload.partitioned_deltas_artifact_ref
    ):
        if update_payload.partitioned_deltas_artifact_ref is None:
            raise AssertionError(
                "partitioned_deltas_artifact_ref unexpectedly missing."
            )
        update_fields["partitioned_deltas_artifact_ref"] = (
            artifact_store.upload_agent_local_json_artifact(
                agent_local_ref=update_payload.partitioned_deltas_artifact_ref,
            )
        )
    if not update_fields:
        return update_payload
    update_fields["delta_format"] = PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED
    return update_payload.model_copy(update=update_fields)


def server_owned_peft_encoder_update_artifact_byte_count(
    *,
    artifact_store: PeftEncoderDeltaArtifactStore,
    update_payload: PeftClassifierDelta,
) -> int:
    """server-owned update artifact ref들의 파일 크기를 합산한다."""

    return artifact_store.server_artifact_refs_byte_count(
        artifact_refs=(
            _adapter_delta_artifact_ref(update_payload),
            update_payload.classifier_head_delta_artifact_ref,
            update_payload.partitioned_deltas_artifact_ref,
        ),
    )


def _adapter_delta_artifact_ref(
    update_payload: PeftClassifierDelta,
) -> str | None:
    return update_payload.peft_adapter_delta_artifact_ref


def _adapter_delta_artifact_ref_field(
    update_payload: PeftClassifierDelta,
) -> str:
    return "peft_adapter_delta_artifact_ref"
