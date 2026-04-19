"""Hydra pseudo-label algorithm preset 해석 helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.src.services.training.query_adaptation.ssl.base import (
    QuerySslAlgorithmConfig,
)


@dataclass(frozen=True, slots=True)
class ResolvedPseudoLabelAlgorithm:
    """실험 config에서 해석한 pseudo-label selection preset."""

    preset_name: str
    acceptance_policy_name: str
    config: QuerySslAlgorithmConfig

    def to_manifest_entry(self) -> dict[str, Any]:
        return {
            "preset_name": self.preset_name,
            "acceptance_policy_name": self.acceptance_policy_name,
            "confidence_threshold": self.config.confidence_threshold,
            "margin_threshold": self.config.margin_threshold,
        }


def resolve_pseudo_label_algorithm(
    cfg: object,
) -> ResolvedPseudoLabelAlgorithm:
    """Hydra config에서 pseudo-label algorithm preset을 해석한다."""

    raw_group = getattr(cfg, "pseudo_label_algorithm", None)
    if raw_group is not None:
        acceptance_policy_name = str(
            getattr(raw_group, "acceptance_policy_name", "") or ""
        ).strip()
        if not acceptance_policy_name:
            raise ValueError(
                "pseudo_label_algorithm.acceptance_policy_name must not be empty."
            )
        preset_name = str(
            getattr(raw_group, "name", "") or acceptance_policy_name
        ).strip()
        return ResolvedPseudoLabelAlgorithm(
            preset_name=preset_name,
            acceptance_policy_name=acceptance_policy_name,
            config=QuerySslAlgorithmConfig(
                confidence_threshold=float(raw_group.confidence_threshold),
                margin_threshold=float(raw_group.margin_threshold),
            ),
        )

    acceptance_policy_name = str(
        getattr(cfg, "pseudo_label_acceptance_policy_name", "") or ""
    ).strip()
    if not acceptance_policy_name:
        raise ValueError(
            "Missing pseudo_label_algorithm config. "
            "Provide pseudo_label_algorithm.<...> fields."
        )
    return ResolvedPseudoLabelAlgorithm(
        preset_name=acceptance_policy_name,
        acceptance_policy_name=acceptance_policy_name,
        config=QuerySslAlgorithmConfig(
            confidence_threshold=float(cfg.pseudo_label_confidence_threshold),
            margin_threshold=float(cfg.pseudo_label_margin_threshold),
        ),
    )


def build_pseudo_label_algorithm_manifest(
    cfg: object,
) -> dict[str, Any] | None:
    """실험 산출물에 남길 pseudo-label algorithm metadata를 만든다."""

    try:
        resolved = resolve_pseudo_label_algorithm(cfg)
    except (AttributeError, TypeError, ValueError):
        return None
    return resolved.to_manifest_entry()


__all__ = [
    "ResolvedPseudoLabelAlgorithm",
    "build_pseudo_label_algorithm_manifest",
    "resolve_pseudo_label_algorithm",
]
