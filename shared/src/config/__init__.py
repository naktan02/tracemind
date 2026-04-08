"""공용 runtime 기본 설정 export.

package import 시 순환 의존을 피하려고 lazy export만 제공한다.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DEFAULT_TRAINING_PROFILE",
    "DEFAULT_TRAINING_TASK_RUNTIME_DEFAULTS",
    "PSEUDO_LABEL_SELF_TRAINING_V1_PROFILE",
    "PSEUDO_LABEL_SELF_TRAINING_V1_TASK_RUNTIME_DEFAULTS",
    "TrainingDefaultsProfile",
    "TrainingTaskRuntimeDefaults",
    "build_default_secure_aggregation_config",
    "build_default_training_objective_config",
    "build_default_training_selection_policy",
]

_EXPORT_TO_MODULE = {
    "DEFAULT_TRAINING_PROFILE": ".training_defaults",
    "DEFAULT_TRAINING_TASK_RUNTIME_DEFAULTS": ".training_default_values",
    "PSEUDO_LABEL_SELF_TRAINING_V1_PROFILE": ".training_defaults",
    "PSEUDO_LABEL_SELF_TRAINING_V1_TASK_RUNTIME_DEFAULTS": ".training_default_values",
    "TrainingDefaultsProfile": ".training_defaults",
    "TrainingTaskRuntimeDefaults": ".training_default_values",
    "build_default_secure_aggregation_config": ".training_defaults",
    "build_default_training_objective_config": ".training_defaults",
    "build_default_training_selection_policy": ".training_defaults",
}


def __getattr__(name: str):
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
