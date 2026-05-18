"""Query-domain weak/strong text view row 해석 helper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

USB_MULTIVIEW_BUILDER_NAME = "usb_multiview"
USB_WEAK_BUILDER_NAME = "usb_weak"


class QuerySslBacktranslationPair(Protocol):
    """원문 하나에서 생성한 두 개의 strong view."""

    aug_0: str
    aug_1: str
    aug_0_pivot_lang: str
    aug_1_pivot_lang: str


def resolve_query_weak_text(row: Mapping[str, Any]) -> str | None:
    """row에서 weak/original view를 읽는다."""

    legacy_value = _optional_row_value(row, "weak_text")
    if legacy_value is not None:
        return legacy_value
    if _optional_row_value(row, "aug_0") or _optional_row_value(row, "aug_1"):
        return _optional_row_value(row, "text")
    return None


def resolve_query_strong_text(row: Mapping[str, Any]) -> str | None:
    """legacy weak/strong 또는 USB aug 후보 중 기본 strong view를 읽는다."""

    legacy_value = _optional_row_value(row, "strong_text")
    if legacy_value is not None:
        return legacy_value
    return _optional_row_value(row, "aug_0") or _optional_row_value(row, "aug_1")


def validate_query_ssl_unlabeled_views(
    *,
    rows: Sequence[LabeledQueryRow],
    view_builder_name: str,
    algorithm_name: str,
) -> None:
    """Query SSL algorithm이 요구하는 unlabeled row view surface를 검증한다."""

    missing_ids = [
        str(row["query_id"])
        for row in rows
        if not row_supports_query_ssl_view_builder(
            row=row,
            view_builder_name=view_builder_name,
        )
    ]
    if missing_ids:
        raise ValueError(
            f"{algorithm_name} requires unlabeled rows compatible with "
            f"{view_builder_name}. Missing examples: {missing_ids[:5]}."
        )


def attach_usb_multiview_candidate_pair(
    row: LabeledQueryRow,
    candidate_pair: QuerySslBacktranslationPair,
) -> LabeledQueryRow:
    """row에 USB strict multiview candidate fields를 붙인다."""

    row_with_views: LabeledQueryRow = dict(row)  # type: ignore[assignment]
    row_with_views["aug_0"] = candidate_pair.aug_0
    row_with_views["aug_1"] = candidate_pair.aug_1
    row_with_views["aug_0_pivot_lang"] = candidate_pair.aug_0_pivot_lang
    row_with_views["aug_1_pivot_lang"] = candidate_pair.aug_1_pivot_lang
    return row_with_views


def rows_have_usb_multiview_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    """모든 row가 strict USB `text/aug_0/aug_1` surface를 갖는지 확인한다."""

    return all(_has_strict_usb_multiview_fields(row) for row in rows)


def validate_usb_multiview_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    """strict USB multiview candidate rows를 검증한다."""

    missing_query_ids = [
        str(row["query_id"])
        for row in rows
        if not _has_strict_usb_multiview_fields(row)
    ]
    if missing_query_ids:
        raise ValueError(
            f"{context} requires non-empty aug_0 and aug_1. "
            f"Missing examples: {missing_query_ids[:5]}."
        )


def validate_usb_weak_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    """USB weak/original view rows를 검증한다."""

    missing_query_ids = [
        str(row["query_id"])
        for row in rows
        if not (
            _optional_row_value(row, "weak_text") or _optional_row_value(row, "text")
        )
    ]
    if missing_query_ids:
        raise ValueError(
            f"{context} requires each unlabeled row to include text or weak_text. "
            f"Missing examples: {missing_query_ids[:5]}."
        )


def row_supports_query_ssl_view_builder(
    *,
    row: Mapping[str, Any],
    view_builder_name: str,
) -> bool:
    """row가 선택된 Query SSL view builder 입력을 만족하는지 반환한다."""

    if view_builder_name == USB_MULTIVIEW_BUILDER_NAME:
        return _has_strict_usb_multiview_fields(row) or _has_legacy_pair_fields(row)
    if view_builder_name == USB_WEAK_BUILDER_NAME:
        return bool(
            _optional_row_value(row, "weak_text") or _optional_row_value(row, "text")
        )
    raise ValueError(f"Unsupported Query SSL view builder: {view_builder_name}.")


def row_supports_weak_strong_pair(row: Mapping[str, Any]) -> bool:
    """agent weak_strong_pair backend가 소비할 수 있는 source row인지 확인한다."""

    return _has_legacy_pair_fields(row) or _has_strict_usb_multiview_fields(row)


def _has_legacy_pair_fields(row: Mapping[str, Any]) -> bool:
    return bool(
        _optional_row_value(row, "weak_text")
        and _optional_row_value(row, "strong_text")
    )


def _has_strict_usb_multiview_fields(row: Mapping[str, Any]) -> bool:
    return bool(
        _optional_row_value(row, "text")
        and _optional_row_value(row, "aug_0")
        and _optional_row_value(row, "aug_1")
    )


def _optional_row_value(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
