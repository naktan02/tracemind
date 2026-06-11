"""Query SSL model forward capability와 auxiliary module 계약."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from torch import Tensor, nn

from methods.ssl.base import QuerySslAlgorithm, TextBatchClassifier


class FeatureReturningTextBatchClassifier(TextBatchClassifier, Protocol):
    """logits path와 dropout 전 pooled feature path를 함께 제공하는 classifier."""

    def extract_pooled_features(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """tokenized text batch의 pooled representation을 반환한다."""


class FeatureMixingTextBatchClassifier(FeatureReturningTextBatchClassifier, Protocol):
    """classifier 직전 feature를 직접 classification할 수 있는 classifier."""

    classifier: nn.Module


def require_pooled_feature_classifier(
    model: TextBatchClassifier,
) -> FeatureReturningTextBatchClassifier:
    """pooled feature capability가 없는 model이면 bootstrap 전에 실패시킨다."""

    extract_pooled_features = getattr(model, "extract_pooled_features", None)
    if not callable(extract_pooled_features):
        raise TypeError(
            "Query SSL model output capability requires extract_pooled_features()."
        )
    return cast(FeatureReturningTextBatchClassifier, model)


def require_feature_mixing_classifier(
    model: TextBatchClassifier,
) -> FeatureMixingTextBatchClassifier:
    """feature-level MixUp에 필요한 classifier head capability를 검증한다."""

    feature_classifier = require_pooled_feature_classifier(model)
    classifier_head = getattr(feature_classifier, "classifier", None)
    if not isinstance(classifier_head, nn.Module):
        raise TypeError(
            "Query SSL feature mixing requires model.classifier to be nn.Module."
        )
    return cast(FeatureMixingTextBatchClassifier, feature_classifier)


def extract_classifier_input_features(
    model: TextBatchClassifier,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """model forward와 같은 classifier 직전 feature를 반환한다."""

    classifier_model = require_feature_mixing_classifier(model)
    pooled_features = classifier_model.extract_pooled_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return prepare_classifier_input_features(classifier_model, pooled_features)


def forward_logits_and_pooled_features_once(
    model: TextBatchClassifier,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """한 번 추출한 pooled feature에서 logits와 feature consumer 출력을 함께 만든다."""

    classifier_model = require_feature_mixing_classifier(model)
    pooled_features = classifier_model.extract_pooled_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    classifier_features = prepare_classifier_input_features(
        classifier_model,
        pooled_features,
    )
    return classifier_model.classifier(classifier_features), pooled_features


def prepare_classifier_input_features(
    model: TextBatchClassifier,
    features: Tensor,
) -> Tensor:
    """pooled feature에 forward path와 같은 classifier 직전 변환을 적용한다."""

    classifier_model = require_feature_mixing_classifier(model)
    dropout = getattr(classifier_model, "dropout", None)
    if isinstance(dropout, nn.Module):
        features = dropout(features)
    classifier_dtype = _module_parameter_dtype(classifier_model.classifier)
    if classifier_dtype is None:
        return features
    return features.to(classifier_dtype)


def classify_classifier_input_features(
    model: TextBatchClassifier,
    features: Tensor,
) -> Tensor:
    """classifier 직전 feature를 classifier head에 직접 통과시킨다."""

    classifier_model = require_feature_mixing_classifier(model)
    classifier_head = classifier_model.classifier
    classifier_dtype = _module_parameter_dtype(classifier_head)
    classifier_features = (
        features if classifier_dtype is None else features.to(classifier_dtype)
    )
    return classifier_head(classifier_features)


def build_query_ssl_auxiliary_modules(
    algorithm: QuerySslAlgorithm,
    *,
    model: TextBatchClassifier,
) -> dict[str, nn.Module]:
    """algorithm이 선택적으로 제공하는 auxiliary trainable module을 만든다."""

    module_builder = getattr(algorithm, "build_auxiliary_modules", None)
    if callable(module_builder):
        return normalize_query_ssl_auxiliary_modules(module_builder(model=model))

    module_mapping = getattr(algorithm, "auxiliary_modules", None)
    if callable(module_mapping):
        module_mapping = module_mapping()
    return normalize_query_ssl_auxiliary_modules(module_mapping)


def normalize_query_ssl_auxiliary_modules(
    modules: object,
) -> dict[str, nn.Module]:
    """checkpoint 가능한 이름 -> nn.Module mapping으로 정규화한다."""

    if modules is None:
        return {}
    if not isinstance(modules, Mapping):
        raise TypeError(
            "Query SSL auxiliary modules must be a mapping of name to nn.Module."
        )
    normalized: dict[str, nn.Module] = {}
    for raw_name, module in modules.items():
        name = _require_auxiliary_module_name(raw_name)
        if not isinstance(module, nn.Module):
            raise TypeError(
                "Query SSL auxiliary module must be nn.Module; "
                f"got {type(module).__name__} for {name!r}."
            )
        normalized[name] = module
    return dict(sorted(normalized.items()))


def query_ssl_auxiliary_trainable_parameters(
    auxiliary_modules: Mapping[str, nn.Module],
) -> tuple[nn.Parameter, ...]:
    """auxiliary module 중 gradient update 대상 parameter만 반환한다."""

    return tuple(
        parameter
        for module in auxiliary_modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def query_ssl_auxiliary_module_state_dicts(
    auxiliary_modules: Mapping[str, nn.Module],
) -> dict[str, dict[str, Any]]:
    """checkpoint payload에 넣을 auxiliary module state dict를 만든다."""

    return {
        name: dict(module.state_dict())
        for name, module in sorted(auxiliary_modules.items())
    }


def load_query_ssl_auxiliary_module_state_dicts(
    auxiliary_modules: Mapping[str, nn.Module],
    state_dicts: object,
) -> None:
    """checkpoint의 auxiliary module state를 현재 module mapping에 복원한다."""

    if state_dicts is None:
        return
    if not isinstance(state_dicts, Mapping):
        raise TypeError("auxiliary_module_state_dicts must be a mapping.")
    if not state_dicts:
        return
    missing = sorted(str(name) for name in set(state_dicts) - set(auxiliary_modules))
    if missing:
        raise ValueError(
            "Checkpoint contains auxiliary modules not provided by current "
            f"algorithm: {missing[:5]}."
        )
    for name, state_dict in sorted(state_dicts.items()):
        module = auxiliary_modules[str(name)]
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Auxiliary module state for {name!r} must be a mapping.")
        module.load_state_dict(state_dict)


def _require_auxiliary_module_name(raw_name: object) -> str:
    name = str(raw_name).strip()
    if not name:
        raise ValueError("Query SSL auxiliary module name must not be empty.")
    return name


def _module_parameter_dtype(module: nn.Module) -> Any:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return None
    return parameter.dtype
