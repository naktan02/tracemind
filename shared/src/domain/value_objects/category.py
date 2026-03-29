"""카테고리 enum."""

from enum import StrEnum


class Category(StrEnum):
    """초기 semantic 카테고리."""

    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    SUICIDAL = "suicidal"
    NORMAL = "normal"
