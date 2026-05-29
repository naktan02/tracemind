"""학습 objective와 selection policy payload 계약."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

TrainingConfigScalar = str | int | float | bool
TrainingConfigInputValue = TrainingConfigScalar | None | Mapping[str, object]

_OBJECTIVE_CONFIG_KEYS = {
    "training_backend_name",
    "algorithm_profile_name",
    "loss",
    "loss_name",
    "confidence_threshold",
    "margin_threshold",
    "example_generation_backend_name",
    "evidence_backend_name",
    "scorer_backend_name",
    "score_policy_name",
    "score_top_k",
    "pseudo_label_algorithm_name",
    "acceptance_policy_name",
    "privacy_guard_name",
}


class TrainingObjectiveConfigPayload(BaseModel):
    """학습 objective 관련 payload.

    `shared`는 backend 기본값을 소유하지 않는다. 이 payload는 runtime이나
    Hydra profile이 선택한 local objective 값을 canonical shape로 정규화한다.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    training_backend_name: str = Field(
        validation_alias=AliasChoices("training_backend_name", "loss"),
        serialization_alias="training_backend_name",
        description="로컬 update backend 식별자.",
    )
    algorithm_profile_name: str | None = Field(
        default=None,
        description="논문/알고리즘 단위 objective 조합 preset 식별자.",
    )
    loss_name: str | None = Field(
        default=None,
        description=(
            "학습 objective의 loss 함수 식별자. backend 선택과는 독립적인 의미 축이다."
        ),
    )
    confidence_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Pseudo-label를 채택하기 위한 최소 confidence.",
    )
    margin_threshold: float | None = Field(
        default=None,
        description="Top1과 top2 score 차이의 최소값.",
    )
    example_generation_backend_name: str | None = Field(
        default=None,
        description="학습 예시 재구성 backend 식별자.",
    )
    evidence_backend_name: str | None = Field(
        default=None,
        description="Pseudo-label evidence 정규화 backend 식별자.",
    )
    scorer_backend_name: str | None = Field(
        default=None,
        description="카테고리 score 계산 backend 식별자.",
    )
    score_policy_name: str | None = Field(
        default=None,
        description="카테고리 내 다중 prototype score 집계 정책 식별자.",
    )
    score_top_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-k score 집계 정책이 사용할 k 값.",
    )
    pseudo_label_algorithm_name: str | None = Field(
        default=None,
        description="Pseudo-label 후보를 평가하는 selection 알고리즘 식별자.",
    )
    acceptance_policy_name: str | None = Field(
        default=None,
        description="Pseudo-label acceptance 정책 식별자.",
    )
    privacy_guard_name: str | None = Field(
        default=None,
        description="로컬 update 보호 계층 식별자.",
    )
    extras: dict[str, TrainingConfigScalar] = Field(
        default_factory=dict,
        description="Objective family별 추가 하이퍼파라미터 확장 슬롯.",
    )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, TrainingConfigInputValue] | None,
    ) -> "TrainingObjectiveConfigPayload":
        """Mapping 입력을 canonical objective config로 정규화한다."""
        if source is None:
            raise ValueError("training_backend_name is required.")
        source = _flatten_objective_mapping(source)
        backend_name = source.get("training_backend_name", source.get("loss"))
        if backend_name is None:
            raise ValueError("training_backend_name is required.")
        pseudo_label_algorithm_name = optional_config_str(
            source.get("pseudo_label_algorithm_name")
        )
        if pseudo_label_algorithm_name is None:
            # compatibility:
            # 과거 objective mapping은 acceptance 정책 이름을 selection 알고리즘
            # 식별자로도 재사용했다. canonical contract는 분리하되,
            # mapping 정규화 경로에서만 얇게 이어받는다.
            pseudo_label_algorithm_name = optional_config_str(
                source.get("acceptance_policy_name")
            )
        return cls(
            training_backend_name=str(backend_name),
            algorithm_profile_name=optional_config_str(
                source.get("algorithm_profile_name")
            ),
            loss_name=optional_config_str(source.get("loss_name")),
            confidence_threshold=optional_config_float(
                source.get("confidence_threshold")
            ),
            margin_threshold=optional_config_float(source.get("margin_threshold")),
            example_generation_backend_name=optional_config_str(
                source.get("example_generation_backend_name")
            ),
            evidence_backend_name=optional_config_str(
                source.get("evidence_backend_name")
            ),
            scorer_backend_name=optional_config_str(source.get("scorer_backend_name")),
            score_policy_name=optional_config_str(source.get("score_policy_name")),
            score_top_k=optional_config_positive_int(source.get("score_top_k")),
            pseudo_label_algorithm_name=pseudo_label_algorithm_name,
            acceptance_policy_name=optional_config_str(
                source.get("acceptance_policy_name")
            ),
            privacy_guard_name=optional_config_str(source.get("privacy_guard_name")),
            extras={
                key: value
                for key, value in source.items()
                if key not in _OBJECTIVE_CONFIG_KEYS
            },
        )

    def to_mapping(self) -> dict[str, TrainingConfigScalar]:
        """canonical objective config를 저장/전송용 flat mapping으로 변환한다."""
        result: dict[str, TrainingConfigScalar] = {
            "training_backend_name": self.training_backend_name
        }
        if self.algorithm_profile_name is not None:
            result["algorithm_profile_name"] = self.algorithm_profile_name
        if self.loss_name is not None:
            result["loss_name"] = self.loss_name
        if self.confidence_threshold is not None:
            result["confidence_threshold"] = self.confidence_threshold
        if self.margin_threshold is not None:
            result["margin_threshold"] = self.margin_threshold
        if self.example_generation_backend_name is not None:
            result["example_generation_backend_name"] = (
                self.example_generation_backend_name
            )
        if self.evidence_backend_name is not None:
            result["evidence_backend_name"] = self.evidence_backend_name
        if self.scorer_backend_name is not None:
            result["scorer_backend_name"] = self.scorer_backend_name
        if self.score_policy_name is not None:
            result["score_policy_name"] = self.score_policy_name
        if self.score_top_k is not None:
            result["score_top_k"] = self.score_top_k
        if self.pseudo_label_algorithm_name is not None:
            result["pseudo_label_algorithm_name"] = self.pseudo_label_algorithm_name
        if self.acceptance_policy_name is not None:
            result["acceptance_policy_name"] = self.acceptance_policy_name
        if self.privacy_guard_name is not None:
            result["privacy_guard_name"] = self.privacy_guard_name
        result.update(self.extras)
        return result

    def get_component_extras(
        self,
        component_scope: str,
        *,
        legacy_keys: Collection[str] = (),
    ) -> dict[str, TrainingConfigScalar]:
        """컴포넌트 scope별 extra 파라미터를 추출한다."""

        normalized_scope = component_scope.strip()
        if not normalized_scope:
            raise ValueError("component_scope must not be empty.")
        prefix = f"{normalized_scope}."
        scoped = {
            key[len(prefix) :]: value
            for key, value in self.extras.items()
            if key.startswith(prefix)
        }
        if scoped:
            return scoped
        return {key: value for key, value in self.extras.items() if key in legacy_keys}

    @property
    def loss(self) -> str:
        """구버전 config key와의 호환을 위한 deprecated alias."""
        return self.training_backend_name


def _flatten_objective_mapping(
    source: Mapping[str, TrainingConfigInputValue],
) -> dict[str, object]:
    """Hydra nested source를 dotted canonical objective key로 낮춘다."""

    result: dict[str, object] = {}
    for key, value in source.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("training objective keys must not be empty.")
        if isinstance(value, Mapping):
            for nested_key, nested_value in _flatten_objective_mapping(value).items():
                result[f"{normalized_key}.{nested_key}"] = nested_value
        else:
            result[normalized_key] = value
    return result


class TrainingSelectionPolicyPayload(BaseModel):
    """로컬 학습 예시 선택 정책 payload."""

    model_config = ConfigDict(extra="forbid")

    max_examples: int | None = Field(
        default=None,
        ge=0,
        description="한 라운드에서 반영할 로컬 예시의 최대 개수.",
    )
    require_feedback: bool | None = Field(
        default=None,
        description="명시적 feedback가 있는 예시만 학습에 반영할지 여부.",
    )
    extras: dict[str, TrainingConfigScalar] = Field(
        default_factory=dict,
        description="Selection policy별 추가 규칙 확장 슬롯.",
    )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, TrainingConfigScalar] | None,
    ) -> "TrainingSelectionPolicyPayload":
        """Mapping 입력을 canonical selection policy로 정규화한다."""
        if source is None:
            return cls()
        return cls(
            max_examples=optional_config_int(source.get("max_examples")),
            require_feedback=optional_config_bool(source.get("require_feedback")),
            extras={
                key: value
                for key, value in source.items()
                if key not in {"max_examples", "require_feedback"}
            },
        )

    def to_mapping(self) -> dict[str, TrainingConfigScalar]:
        """canonical selection policy를 저장/전송용 flat mapping으로 변환한다."""
        result: dict[str, TrainingConfigScalar] = {}
        if self.max_examples is not None:
            result["max_examples"] = self.max_examples
        if self.require_feedback is not None:
            result["require_feedback"] = self.require_feedback
        result.update(self.extras)
        return result


def optional_config_float(value: TrainingConfigScalar | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Expected float-like config value, got bool.")
    return float(value)


def optional_config_int(value: TrainingConfigScalar | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Expected int-like config value, got bool.")
    return int(value)


def optional_config_bool(value: TrainingConfigScalar | None) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("Expected bool config value.")
    return value


def optional_config_str(value: TrainingConfigScalar | None) -> str | None:
    if value is None:
        return None
    return str(value)


def optional_config_positive_int(value: TrainingConfigScalar | None) -> int | None:
    parsed = optional_config_int(value)
    if parsed is None:
        return None
    if parsed < 1:
        raise ValueError("Expected positive int config value.")
    return parsed


TrainingObjectiveConfig = TrainingObjectiveConfigPayload
TrainingSelectionPolicy = TrainingSelectionPolicyPayload
