"""Artifact-oriented domain entities."""

from shared.src.contracts.model_contracts import ModelManifest

from .labeled_query import LabeledQuery, LabeledQuerySet
from .prototype_pack import (
    CategoryPrototype,
    PrototypePack,
    SingleCategoryPrototype,
    SinglePrototypePack,
)

__all__ = [
    "CategoryPrototype",
    "LabeledQuery",
    "LabeledQuerySet",
    "ModelManifest",
    "PrototypePack",
    "SingleCategoryPrototype",
    "SinglePrototypePack",
]
