"""SoftMatch upstream reference metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SOFTMATCH_ORIGINAL_REPOSITORY = "https://github.com/microsoft/Semi-supervised-learning"
SOFTMATCH_ORIGINAL_COMMIT = "1ef4cbebcc0b368158315aeb425053858cf6c845"
SOFTMATCH_ORIGINAL_ALGORITHM_PATH = "semilearn/algorithms/softmatch/softmatch.py"
SOFTMATCH_ORIGINAL_UTILS_PATH = "semilearn/algorithms/softmatch/utils.py"


def softmatch_original_parameter_mapping(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """TraceMind parameter 이름을 USB SoftMatch 원본 인자 이름으로 대응한다."""

    return {
        "T": float(parameters["temperature"]),
        "hard_label": bool(parameters.get("hard_label", True)),
        "dist_align": bool(parameters.get("dist_align", True)),
        "dist_uniform": bool(parameters.get("dist_uniform", True)),
        "ema_p": float(parameters.get("ema_p", 0.999)),
        "n_sigma": float(parameters.get("n_sigma", 2.0)),
        "per_class": bool(parameters.get("per_class", False)),
    }
