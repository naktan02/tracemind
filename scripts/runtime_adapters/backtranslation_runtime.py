"""query SSL augmentation이 쓰는 backtranslation runtime bridge."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from methods.adaptation.query_text_views.unlabeled_preparation import (
    QuerySslCandidatePairBuilder,
)


def build_nllb_backtranslation_candidate_pairs_from_params(
    *,
    texts: Sequence[str],
    source_lang: str,
    pivot_languages: tuple[str, str],
    model_id: str,
    revision: str,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    torch_dtype: str,
    cache_dir: str | None,
    local_files_only: bool,
) -> Any:
    """명시 파라미터로 NLLB candidate pair를 생성한다."""

    from agent.src.services.language.backtranslation_service import (
        NllbBacktranslationService,
    )

    service = NllbBacktranslationService(
        source_lang=source_lang,
        pivot_languages=pivot_languages,
        model_id=model_id,
        revision=revision,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return service.build_candidate_pairs(texts=list(texts))


def build_nllb_backtranslation_candidate_pairs(
    cfg: Any,
    *,
    texts: list[str],
) -> Any:
    """Hydra augmenter config로 NLLB candidate pair를 생성한다."""

    return build_nllb_backtranslation_candidate_pairs_from_params(
        texts=texts,
        source_lang=str(cfg.query_ssl_augmenter.source_lang),
        pivot_languages=tuple(
            str(language) for language in cfg.query_ssl_augmenter.pivot_languages
        ),
        model_id=str(cfg.query_ssl_augmenter.model_id),
        revision=str(cfg.query_ssl_augmenter.revision),
        device=str(cfg.query_ssl_augmenter.device),
        batch_size=int(cfg.query_ssl_augmenter.batch_size),
        max_new_tokens=int(cfg.query_ssl_augmenter.max_new_tokens),
        torch_dtype=str(getattr(cfg.query_ssl_augmenter, "torch_dtype", "auto")),
        cache_dir=str(cfg.query_ssl_augmenter.cache_dir),
        local_files_only=bool(cfg.query_ssl_augmenter.local_files_only),
    )


def build_nllb_backtranslation_candidate_pair_builder(
    cfg: Any,
) -> QuerySslCandidatePairBuilder:
    """Hydra cfg로 Query SSL NLLB candidate pair builder를 만든다."""

    def _build(texts: Sequence[str]) -> Any:
        return build_nllb_backtranslation_candidate_pairs(cfg, texts=list(texts))

    return _build
