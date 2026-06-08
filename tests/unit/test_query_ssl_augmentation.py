from __future__ import annotations

from methods.adaptation.query_text_views.unlabeled_preparation import (
    NLLB_BACKTRANSLATION_AUGMENTER,
    QuerySslAugmenterSettings,
    prepare_query_ssl_unlabeled_rows,
)
from methods.adaptation.query_text_views.view_rows import (
    USB_MULTIVIEW_BUILDER_NAME,
    USB_WEAK_STRONG_PAIR_BUILDER_NAME,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def _augmenter_settings(cache_dir: str) -> QuerySslAugmenterSettings:
    return QuerySslAugmenterSettings(
        name="backtranslation_nllb_en_de_fr_usb_v1",
        augmenter_type=NLLB_BACKTRANSLATION_AUGMENTER,
        source_lang="eng_Latn",
        pivot_languages=("deu_Latn", "fra_Latn"),
        model_id="facebook/nllb-200-distilled-600M",
        revision="main",
        device="cpu",
        local_files_only=True,
        batch_size=8,
        max_new_tokens=256,
        torch_dtype="auto",
        cache_dir=cache_dir,
    )


def _row(query_id: str, text: str) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=text,
        raw_label_scheme="agent_local_unlabeled",
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
    tmp_path,
) -> None:
    settings = _augmenter_settings(str(tmp_path))
    rows = [_row("u1", "I feel anxious today.")]

    prepared = prepare_query_ssl_unlabeled_rows(
        view_builder_name=USB_MULTIVIEW_BUILDER_NAME,
        algorithm_name="fixmatch",
        rows=rows,
        source_jsonl=None,
        augmenter_settings=settings,
        candidate_pair_builder=lambda texts: (
            _FakeBacktranslationAugmenter().build_candidate_pairs(texts=texts)
        ),
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

    cached = prepare_query_ssl_unlabeled_rows(
        view_builder_name=USB_MULTIVIEW_BUILDER_NAME,
        algorithm_name="fixmatch",
        rows=rows,
        source_jsonl=None,
        augmenter_settings=settings,
        candidate_pair_builder=lambda _texts: (_ for _ in ()).throw(
            AssertionError("cache hit should skip regeneration")
        ),
    )

    assert cached.mode == "cache_hit"
    assert cached.cache_hit is True
    assert cached.rows[0]["aug_0"] == "de::I feel anxious today."


def test_prepare_weak_strong_pair_unlabeled_rows_reuses_usb_candidate_preparation(
    tmp_path,
) -> None:
    settings = _augmenter_settings(str(tmp_path))
    rows = [_row("u1", "I feel anxious today.")]

    prepared = prepare_query_ssl_unlabeled_rows(
        view_builder_name=USB_WEAK_STRONG_PAIR_BUILDER_NAME,
        algorithm_name="comatch",
        rows=rows,
        source_jsonl=None,
        augmenter_settings=settings,
        candidate_pair_builder=lambda texts: (
            _FakeBacktranslationAugmenter().build_candidate_pairs(texts=texts)
        ),
    )

    assert prepared.uses_strong_view_candidates is True
    assert prepared.rows[0]["aug_0"] == "de::I feel anxious today."
    assert prepared.rows[0]["aug_1"] == "fr::I feel anxious today."
