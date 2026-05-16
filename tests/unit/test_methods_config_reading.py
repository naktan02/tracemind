"""methods common config parsing helper tests."""

from __future__ import annotations

import pytest

from methods.common.config_reading import (
    freeze_mapping,
    read_float,
    read_str,
    read_str_tuple,
    read_unit_interval_float,
    validate_allowed_keys,
)


def test_validate_allowed_keys_reports_unknown_keys() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unsupported example config key\(s\): \['unknown'\]\.",
    ):
        validate_allowed_keys(
            {"known": 1, "unknown": 2},
            allowed_keys={"known"},
            config_name="example config",
        )


def test_read_float_rejects_bool_with_field_prefix() -> None:
    with pytest.raises(ValueError, match="profile.rate must be a number."):
        read_float({"rate": True}, "rate", field_prefix="profile")


def test_read_unit_interval_float_preserves_lora_range_message() -> None:
    with pytest.raises(ValueError, match="dropout must be between 0 and 1."):
        read_unit_interval_float({"dropout": 1.5}, "dropout", 0.1)


def test_read_str_tuple_accepts_comma_string_and_sequence() -> None:
    assert read_str_tuple({"labels": "a, b,,c"}, "labels", ()) == ("a", "b", "c")
    assert read_str_tuple({"labels": ["a,b", " c "]}, "labels", ()) == (
        "a",
        "b",
        "c",
    )


def test_freeze_mapping_returns_immutable_copy() -> None:
    source = {"alpha": 1}
    frozen = freeze_mapping(source)
    source["alpha"] = 2

    assert frozen["alpha"] == 1
    with pytest.raises(TypeError):
        frozen["alpha"] = 3  # type: ignore[index]


def test_read_str_trims_and_rejects_empty() -> None:
    assert read_str({"name": " demo "}, "name") == "demo"
    with pytest.raises(ValueError, match="profile.name must not be empty."):
        read_str({"name": " "}, "name", field_prefix="profile")
