"""Legacy classifier-head scoring import path."""

# ruff: noqa: F401

from methods.adaptation.text_classifier.feature_head.scoring import (
    CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    CLASSIFIER_HEAD_LOGITS_CONFIDENCE_KIND,
    CLASSIFIER_HEAD_LOGITS_SCORING_BACKEND_CATALOG_ENTRY,
    ClassifierHeadLogitsScoringBackend,
    build_classifier_head_logits_scoring_backend,
)
