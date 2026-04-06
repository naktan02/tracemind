"""Shared adapter 상태/업데이트 payload와 직렬화 유틸리티.

이 모듈은 "서버와 agent 사이에서 어떤 adapter 상태/업데이트를 주고받는가"를
정의하는 source of truth다.

현재 concrete 구현은 `diagonal_scale` 하나뿐이며, 의미는 다음과 같다.

1. state payload
   - 현재 라운드에서 모든 agent가 공통으로 적용하는 전역 shared adapter 상태
2. update payload
   - 개별 agent가 로컬 학습 후 서버로 올리는 shared adapter 업데이트
3. diagonal scale family
   - 임베딩 각 차원에 곱하는 scale 벡터를 공유하는 adapter family
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SharedAdapterStatePayload(BaseModel):
    """전역 shared adapter 상태 payload 공통 필드.

    필드 의미:

    - `schema_version`: payload contract 버전
    - `adapter_kind`: 어떤 adapter family인지 식별하는 discriminator
    - `model_id`: 이 adapter가 붙는 backbone/model 식별자
    - `model_revision`: 서버가 현재 배포 중인 shared adapter revision
    - `training_scope`: `adapter_only` 같은 학습 범위 식별자
    - `updated_at`: 서버가 이 상태를 발행한 시각
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(description="Payload contract 버전.")
    adapter_kind: str = Field(
        default="diagonal_scale",
        description="Adapter family discriminator. 현재는 diagonal_scale만 지원한다.",
    )
    model_id: str = Field(description="이 adapter가 결합되는 backbone/model 식별자.")
    model_revision: str = Field(description="서버가 발행한 shared adapter revision.")
    training_scope: str = Field(description="이 상태가 적용되는 학습 범위 식별자.")
    updated_at: datetime = Field(description="서버가 이 상태를 기록한 UTC 시각.")


class DiagonalScaleAdapterStatePayload(SharedAdapterStatePayload):
    """현재 concrete 구현인 diagonal scale adapter 상태 payload.

    `dimension_scales[j]`는 임베딩 j번째 차원에 곱하는 전역 scale 값이다.
    runtime 적용식은 `x' = normalize(x ⊙ s)`다.
    """

    dimension_scales: list[float] = Field(
        description=(
            "임베딩 차원별 전역 scale 벡터. j번째 값은 j번째 차원을 얼마나 "
            "늘이거나 줄일지 나타낸다."
        )
    )


class SharedAdapterUpdatePayload(BaseModel):
    """로컬 학습이 생성한 shared adapter update payload 공통 필드.

    필드 의미:

    - `schema_version`: payload contract 버전
    - `adapter_kind`: 어떤 adapter family의 update인지 식별자
    - `model_id`: 이 update가 대상으로 삼는 backbone/model 식별자
    - `base_model_revision`: 어떤 전역 revision을 기준으로 계산한 update인지
    - `training_scope`: `adapter_only` 같은 학습 범위 식별자
    - `example_count`: update 계산에 실제 반영된 로컬 예시 수
    - `created_at`: agent가 update를 기록한 시각
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(description="Payload contract 버전.")
    adapter_kind: str = Field(
        default="diagonal_scale",
        description="Adapter family discriminator. 현재는 diagonal_scale만 지원한다.",
    )
    model_id: str = Field(
        description="이 update가 대상으로 삼는 backbone/model 식별자."
    )
    base_model_revision: str = Field(
        description="로컬 update가 계산된 기준 shared adapter revision."
    )
    training_scope: str = Field(description="이 update가 속한 학습 범위 식별자.")
    example_count: int = Field(
        ge=0,
        description="실제 update 계산에 반영된 로컬 예시 수.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Agent가 이 update payload를 생성한 UTC 시각.",
    )


class DiagonalScaleAdapterUpdatePayload(SharedAdapterUpdatePayload):
    """현재 concrete 구현인 diagonal scale adapter update payload.

    `dimension_deltas[j]`는 j번째 차원 scale에 더할 변화량이다.
    현재 heuristic 구현에서는 accepted example 임베딩 평균 방향에서 유도된
    전역 차원 보정량이며, 특정 prototype 좌표로 직접 이동시키는 값은 아니다.
    """

    dimension_deltas: list[float] = Field(
        description="차원별 scale에 더할 delta 벡터."
    )
    mean_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Update에 반영된 accepted example들의 평균 confidence.",
    )
    mean_margin: float | None = Field(
        default=None,
        description="Accepted example들의 평균 top1-top2 margin.",
    )
    label_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Accepted example의 pseudo-label 분포. drift 관찰용 메타데이터다.",
    )


VectorAdapterStatePayload = DiagonalScaleAdapterStatePayload
VectorAdapterDeltaPayload = DiagonalScaleAdapterUpdatePayload

_STATE_PAYLOAD_TYPES: dict[str, type[SharedAdapterStatePayload]] = {
    "diagonal_scale": DiagonalScaleAdapterStatePayload,
}
_UPDATE_PAYLOAD_TYPES: dict[str, type[SharedAdapterUpdatePayload]] = {
    "diagonal_scale": DiagonalScaleAdapterUpdatePayload,
}


def _dump_payload(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _load_payload_data(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_shared_adapter_state_payload(path: Path) -> SharedAdapterStatePayload:
    """JSON 파일에서 shared adapter state payload를 읽는다."""
    data = _load_payload_data(path)
    adapter_kind = str(data.get("adapter_kind", "diagonal_scale"))
    payload_type = _STATE_PAYLOAD_TYPES.get(adapter_kind)
    if payload_type is None:
        raise ValueError(f"Unsupported shared adapter state kind: {adapter_kind}")
    return payload_type.model_validate(data)


def dump_shared_adapter_state_payload(
    path: Path,
    payload: SharedAdapterStatePayload,
) -> None:
    """shared adapter state payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_shared_adapter_update_payload(path: Path) -> SharedAdapterUpdatePayload:
    """JSON 파일에서 shared adapter update payload를 읽는다."""
    data = _load_payload_data(path)
    adapter_kind = str(data.get("adapter_kind", "diagonal_scale"))
    payload_type = _UPDATE_PAYLOAD_TYPES.get(adapter_kind)
    if payload_type is None:
        raise ValueError(f"Unsupported shared adapter update kind: {adapter_kind}")
    return payload_type.model_validate(data)


def dump_shared_adapter_update_payload(
    path: Path,
    payload: SharedAdapterUpdatePayload,
) -> None:
    """shared adapter update payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_vector_adapter_state_payload(path: Path) -> VectorAdapterStatePayload:
    """JSON 파일에서 diagonal scale adapter state payload를 읽는다."""
    payload = load_shared_adapter_state_payload(path)
    if not isinstance(payload, DiagonalScaleAdapterStatePayload):
        raise ValueError(
            "Expected diagonal_scale adapter state payload, "
            f"got {payload.adapter_kind}."
        )
    return payload


def dump_vector_adapter_state_payload(
    path: Path,
    payload: VectorAdapterStatePayload,
) -> None:
    """diagonal scale adapter state payload를 JSON 파일로 기록한다."""
    dump_shared_adapter_state_payload(path, payload)


def load_vector_adapter_delta_payload(path: Path) -> VectorAdapterDeltaPayload:
    """JSON 파일에서 diagonal scale adapter update payload를 읽는다."""
    payload = load_shared_adapter_update_payload(path)
    if not isinstance(payload, DiagonalScaleAdapterUpdatePayload):
        raise ValueError(
            "Expected diagonal_scale adapter update payload, "
            f"got {payload.adapter_kind}."
        )
    return payload


def dump_vector_adapter_delta_payload(
    path: Path,
    payload: VectorAdapterDeltaPayload,
) -> None:
    """diagonal scale adapter update payload를 JSON 파일로 기록한다."""
    dump_shared_adapter_update_payload(path, payload)


# ------------------------------------------------------------------ #
# Factory 함수                                                         #
# ------------------------------------------------------------------ #


def make_identity_state_payload(
    *,
    model_id: str,
    model_revision: str,
    embedding_dim: int,
    training_scope: str = "adapter_only",
    updated_at: datetime | None = None,
) -> DiagonalScaleAdapterStatePayload:
    """모든 차원 scale=1.0 인 identity(단위) adapter state payload를 만든다.

    새 round 시작 또는 테스트 fixture 생성에 사용한다.

    >>> state = make_identity_state_payload(
    ...     model_id="bg-m3", model_revision="rev_001", embedding_dim=768
    ... )
    """
    return DiagonalScaleAdapterStatePayload(
        schema_version="shared_adapter_state.v1",
        adapter_kind="diagonal_scale",
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        updated_at=updated_at or datetime.now(tz=timezone.utc),
        dimension_scales=[1.0] * embedding_dim,
    )


def make_diagonal_delta_payload(
    *,
    model_id: str,
    base_model_revision: str,
    dimension_deltas: list[float],
    example_count: int,
    mean_confidence: float,
    training_scope: str = "adapter_only",
    mean_margin: float | None = None,
    label_counts: dict[str, int] | None = None,
    created_at: datetime | None = None,
) -> DiagonalScaleAdapterUpdatePayload:
    """diagonal scale adapter update payload를 만드는 표준 factory.

    >>> delta = make_diagonal_delta_payload(
    ...     model_id="bg-m3",
    ...     base_model_revision="rev_001",
    ...     dimension_deltas=[0.01] * 768,
    ...     example_count=10,
    ...     mean_confidence=0.85,
    ... )
    """
    return DiagonalScaleAdapterUpdatePayload(
        schema_version="vector_adapter_delta.v1",
        adapter_kind="diagonal_scale",
        model_id=model_id,
        base_model_revision=base_model_revision,
        training_scope=training_scope,
        example_count=example_count,
        created_at=created_at or datetime.now(tz=timezone.utc),
        dimension_deltas=dimension_deltas,
        mean_confidence=mean_confidence,
        mean_margin=mean_margin,
        label_counts=label_counts or {},
    )
