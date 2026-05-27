from __future__ import annotations

from omegaconf import OmegaConf

from scripts.experiments.query_peft_ssl.query_ssl.augmentation import (
    prepare_usb_multiview_unlabeled_rows,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def _cfg(cache_dir: str) -> object:
    return OmegaConf.create(
        {
            "query_ssl_method": {
                "algorithm_name": "fixmatch",
            },
            "query_ssl_augmenter": {
                "name": "backtranslation_nllb_en_de_fr_usb_v1",
                "augmenter_type": "nllb_backtranslation",
                "source_lang": "eng_Latn",
                "pivot_languages": ["deu_Latn", "fra_Latn"],
                "model_id": "facebook/nllb-200-distilled-600M",
                "revision": "main",
                "device": "cpu",
                "local_files_only": True,
                "batch_size": 8,
                "max_new_tokens": 256,
                "torch_dtype": "auto",
                "cache_dir": cache_dir,
            },
        }
    )


def _row(query_id: str, text: str) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=text,
        raw_label_scheme="query_buffer",
        raw_label="unknown",
        mapped_label_4="anxiety",
        locale="en-US",
        annotation_source="query_ssl_raw",
        approved_by=None,
        created_at="2026-04-19T00:00:00+00:00",
    )


class _FakeBacktranslationAugmenter:
    def build_candidate_pairs(self, *, texts):
        return [
            type(
                "_Pair",
                (),
                {
                    "aug_0": f"de::{text}",
                    "aug_1": f"fr::{text}",
                    "aug_0_pivot_lang": "deu_Latn",
                    "aug_1_pivot_lang": "fra_Latn",
                },
            )()
            for text in texts
        ]


def test_prepare_usb_multiview_unlabeled_rows_generates_and_caches(
    monkeypatch,
    tmp_path,
) -> None:
    cfg = _cfg(str(tmp_path))
    rows = [_row("u1", "I feel anxious today.")]

    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.query_ssl.augmentation."
        "build_nllb_backtranslation_candidate_pairs",
        lambda _cfg, *, texts: _FakeBacktranslationAugmenter().build_candidate_pairs(
            texts=texts
        ),
    )

    prepared = prepare_usb_multiview_unlabeled_rows(
        cfg,
        rows=rows,
        source_jsonl=None,
        algorithm_name="fixmatch",
    )

    assert prepared.mode == "generated_and_cached"
    assert prepared.cache_hit is False
    assert prepared.rows[0]["aug_0"] == "de::I feel anxious today."
    assert prepared.rows[0]["aug_1"] == "fr::I feel anxious today."
    assert prepared.prepared_jsonl_path is not None
    assert prepared.prepared_jsonl_path.exists()
    assert prepared.manifest_path is not None
    assert prepared.manifest_path.exists()
    assert prepared.summary_path is not None
    assert prepared.summary_path.exists()

    monkeypatch.setattr(
        "scripts.experiments.query_peft_ssl.query_ssl.augmentation."
        "build_nllb_backtranslation_candidate_pairs",
        lambda _cfg, *, texts: (_ for _ in ()).throw(
            AssertionError("cache hit should skip regeneration")
        ),
    )

    cached = prepare_usb_multiview_unlabeled_rows(
        cfg,
        rows=rows,
        source_jsonl=None,
        algorithm_name="fixmatch",
    )

    assert cached.mode == "cache_hit"
    assert cached.cache_hit is True
    assert cached.rows[0]["aug_0"] == "de::I feel anxious today."
