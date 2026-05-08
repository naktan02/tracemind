"""Server aggregation artifact reference helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AggregatedArtifactRefSet:
    """Aggregation 결과가 가리키는 server-owned artifact refs."""

    lora_adapter_artifact_ref: str
    classifier_head_artifact_ref: str
    artifact_format: str


@dataclass(frozen=True, slots=True)
class AggregatedArtifactRefBuilder:
    """next revision 기준 server-owned artifact ref를 만든다."""

    artifact_ref_prefix: str
    artifact_format: str

    def __post_init__(self) -> None:
        _require_non_empty_str(
            self.artifact_ref_prefix,
            field_name="artifact_ref_prefix",
        )
        _require_non_empty_str(self.artifact_format, field_name="artifact_format")

    def build_lora_classifier_refs(
        self,
        *,
        next_model_revision: str,
    ) -> AggregatedArtifactRefSet:
        """LoRA adapter와 classifier head aggregate artifact refs를 만든다."""

        return AggregatedArtifactRefSet(
            lora_adapter_artifact_ref=self.build_ref(
                next_model_revision=next_model_revision,
                artifact_name="lora_adapter",
            ),
            classifier_head_artifact_ref=self.build_ref(
                next_model_revision=next_model_revision,
                artifact_name="classifier_head",
            ),
            artifact_format=self.artifact_format,
        )

    def build_ref(
        self,
        *,
        next_model_revision: str,
        artifact_name: str,
    ) -> str:
        """prefix/revision/name으로 opaque server-owned ref를 만든다."""

        return "/".join(
            (
                self.artifact_ref_prefix.rstrip("/"),
                _slug_ref_part(next_model_revision),
                _slug_ref_part(artifact_name),
            )
        )


def _slug_ref_part(value: str) -> str:
    normalized = value.strip().replace("/", "_")
    if not normalized:
        raise ValueError("artifact ref path parts must not be empty.")
    return normalized


def _require_non_empty_str(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")
