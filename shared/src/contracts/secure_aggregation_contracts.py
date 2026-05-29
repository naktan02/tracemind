"""Secure aggregation/encryption 학습 payload 계약."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.src.contracts.training_objective_contracts import (
    TrainingConfigScalar,
    optional_config_bool,
    optional_config_str,
)


class SecureAggregationConfigPayload(BaseModel):
    """학습 task가 요구하는 secure aggregation/encryption 설정."""

    model_config = ConfigDict(extra="forbid")

    required: bool = Field(
        default=False,
        description="이번 task에서 secure aggregation이 필수인지 여부.",
    )
    aggregation_backend_name: str | None = Field(
        default=None,
        description="secure aggregation backend 식별자.",
    )
    encryption_scheme_name: str | None = Field(
        default=None,
        description="예: ckks, paillier 같은 encryption scheme 식별자.",
    )
    key_ref: str | None = Field(
        default=None,
        description="암호화 컨텍스트/공개키 material 참조값.",
    )
    ciphertext_format: str | None = Field(
        default=None,
        description="예: ckks_vector_v1 같은 ciphertext 직렬화 포맷 식별자.",
    )
    extras: dict[str, TrainingConfigScalar] = Field(
        default_factory=dict,
        description="secure aggregation backend별 추가 파라미터 확장 슬롯.",
    )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, TrainingConfigScalar] | None,
    ) -> "SecureAggregationConfigPayload":
        if source is None:
            return cls()
        return cls(
            required=optional_config_bool(source.get("required")) or False,
            aggregation_backend_name=optional_config_str(
                source.get("aggregation_backend_name")
            ),
            encryption_scheme_name=optional_config_str(
                source.get("encryption_scheme_name")
            ),
            key_ref=optional_config_str(source.get("key_ref")),
            ciphertext_format=optional_config_str(source.get("ciphertext_format")),
            extras={
                key: value
                for key, value in source.items()
                if key
                not in {
                    "required",
                    "aggregation_backend_name",
                    "encryption_scheme_name",
                    "key_ref",
                    "ciphertext_format",
                }
            },
        )

    @model_validator(mode="after")
    def _normalize_required(self) -> "SecureAggregationConfigPayload":
        if self.required:
            return self
        if (
            any(
                value is not None
                for value in (
                    self.aggregation_backend_name,
                    self.encryption_scheme_name,
                    self.key_ref,
                    self.ciphertext_format,
                )
            )
            or self.extras
        ):
            self.required = True
        return self

    def to_mapping(self) -> dict[str, TrainingConfigScalar]:
        result: dict[str, TrainingConfigScalar] = {"required": self.required}
        if self.aggregation_backend_name is not None:
            result["aggregation_backend_name"] = self.aggregation_backend_name
        if self.encryption_scheme_name is not None:
            result["encryption_scheme_name"] = self.encryption_scheme_name
        if self.key_ref is not None:
            result["key_ref"] = self.key_ref
        if self.ciphertext_format is not None:
            result["ciphertext_format"] = self.ciphertext_format
        result.update(self.extras)
        return result


class SecureAggregationSubmissionPayload(BaseModel):
    """업로드된 update가 어떤 secure aggregation/encryption metadata를 따르는지."""

    model_config = ConfigDict(extra="forbid")

    aggregation_backend_name: str = Field(
        description="제출된 암호문/secure aggregation backend 식별자."
    )
    encryption_scheme_name: str | None = Field(
        default=None,
        description="예: ckks, paillier 같은 encryption scheme 식별자.",
    )
    key_ref: str | None = Field(
        default=None,
        description="사용한 키 material 또는 encryption context 참조값.",
    )
    ciphertext_format: str | None = Field(
        default=None,
        description="업로드 payload의 ciphertext 직렬화 포맷 식별자.",
    )
    extras: dict[str, TrainingConfigScalar] = Field(
        default_factory=dict,
        description="secure submission별 추가 메타데이터 확장 슬롯.",
    )


SecureAggregationConfig = SecureAggregationConfigPayload
SecureAggregationSubmission = SecureAggregationSubmissionPayload
