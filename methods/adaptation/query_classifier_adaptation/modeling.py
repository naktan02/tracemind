"""Query adaptation의 기존 import 경로를 유지하는 compatibility shim."""

from __future__ import annotations

from methods.adaptation.lora_classifier.modeling import (
    LoraTextClassifier as LoraTextClassifier,
)
from methods.adaptation.lora_classifier.modeling import (
    build_model as build_model,
)
from methods.adaptation.lora_classifier.modeling import (
    count_parameters as count_parameters,
)
from methods.adaptation.lora_classifier.modeling import (
    require_transformer_stack as require_transformer_stack,
)
