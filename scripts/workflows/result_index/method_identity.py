"""Result index에서 실험 방법론 identity를 canonical하게 해석한다."""

from __future__ import annotations


def normalize_ssl_method_name(value: str | None) -> str | None:
    """Hydra preset leaf를 사람이 비교하는 SSL 방법론 이름으로 낮춘다."""

    method_name = _clean(value)
    if method_name is None:
        return None
    for suffix in ("_usb_v1", "_classic_v1", "_v1"):
        if method_name.endswith(suffix):
            return method_name[: -len(suffix)]
    return method_name


def is_ssl_method_preset(value: str | None) -> bool:
    method_name = _clean(value)
    if method_name is None:
        return False
    return method_name.endswith(("_usb_v1", "_classic_v1", "_v1"))


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
