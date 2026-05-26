"""Compatibility shim for legacy lora_classifier training loop imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.training.loops import (
    build_optimizer,
    evaluate_classifier,
    set_seed,
    train_classifier,
    train_query_ssl_classifier,
    trainable_model_parameters,
)
