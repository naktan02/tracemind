"""query SSL augmentation이 쓰는 backtranslation runtime bridge."""

from __future__ import annotations

from typing import Any


def build_nllb_backtranslation_candidate_pairs(
    cfg: Any,
    *,
    texts: list[str],
) -> Any:
    """Hydra augmenter config로 NLLB candidate pair를 생성한다."""

    from agent.src.services.language.backtranslation_service import (
        NllbBacktranslationService,
    )

    service = NllbBacktranslationService(
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
    return service.build_candidate_pairs(texts=texts)
