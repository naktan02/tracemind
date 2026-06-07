"""family_extension용 TypeScript wellbeing contract type generator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

import agent.src.contracts.captured_text_contracts as captured_text_contracts
import agent.src.contracts.child_support_contracts as child_support_contracts
import agent.src.contracts.family_access_contracts as family_access_contracts
import agent.src.contracts.typing_segment_contracts as typing_segment_contracts
import agent.src.contracts.wellbeing_signal_contracts as wellbeing_signal_contracts

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "apps/family_extension/src/contracts/generated.ts"

TYPE_SOURCES: tuple[tuple[str, Any], ...] = (
    ("FamilyAccessMode", family_access_contracts.FamilyAccessMode),
    ("FamilyAccessRole", family_access_contracts.FamilyAccessRole),
    (
        "FamilySetupStatusPayload",
        family_access_contracts.FamilySetupStatusPayload,
    ),
    (
        "FamilySetupRequestPayload",
        family_access_contracts.FamilySetupRequestPayload,
    ),
    (
        "FamilySetupResponsePayload",
        family_access_contracts.FamilySetupResponsePayload,
    ),
    (
        "FamilyUnlockRequestPayload",
        family_access_contracts.FamilyUnlockRequestPayload,
    ),
    (
        "FamilyUnlockResponsePayload",
        family_access_contracts.FamilyUnlockResponsePayload,
    ),
    (
        "ChildSupportAssistantMode",
        child_support_contracts.ChildSupportAssistantMode,
    ),
    (
        "ChildSupportSafetyLevel",
        child_support_contracts.ChildSupportSafetyLevel,
    ),
    (
        "ChildSupportScopeStatus",
        child_support_contracts.ChildSupportScopeStatus,
    ),
    (
        "ChildSupportSuggestionPayload",
        child_support_contracts.ChildSupportSuggestionPayload,
    ),
    (
        "ChildSupportConversationRequestPayload",
        child_support_contracts.ChildSupportConversationRequestPayload,
    ),
    (
        "ChildSupportConversationResponsePayload",
        child_support_contracts.ChildSupportConversationResponsePayload,
    ),
    (
        "ChildSupportProactivePromptPayload",
        child_support_contracts.ChildSupportProactivePromptPayload,
    ),
    ("WellbeingSignalLevel", wellbeing_signal_contracts.WellbeingSignalLevel),
    ("WellbeingSignalTrend", wellbeing_signal_contracts.WellbeingSignalTrend),
    (
        "WellbeingSignalConfidence",
        wellbeing_signal_contracts.WellbeingSignalConfidence,
    ),
    ("WellbeingSignalRange", wellbeing_signal_contracts.WellbeingSignalRange),
    (
        "WellbeingSignalSummaryPayload",
        wellbeing_signal_contracts.WellbeingSignalSummaryPayload,
    ),
    (
        "WellbeingSignalTimeseriesPointPayload",
        wellbeing_signal_contracts.WellbeingSignalTimeseriesPointPayload,
    ),
    (
        "WellbeingSignalTimeseriesPayload",
        wellbeing_signal_contracts.WellbeingSignalTimeseriesPayload,
    ),
    (
        "ParentUnlockRequestPayload",
        wellbeing_signal_contracts.ParentUnlockRequestPayload,
    ),
    (
        "ParentUnlockResponsePayload",
        wellbeing_signal_contracts.ParentUnlockResponsePayload,
    ),
    (
        "TypingSegmentSourceType",
        typing_segment_contracts.TypingSegmentSourceType,
    ),
    ("TypingSurfaceType", typing_segment_contracts.TypingSurfaceType),
    (
        "TypingCaptureConfidence",
        typing_segment_contracts.TypingCaptureConfidence,
    ),
    (
        "TypingSegmentStatsPayload",
        typing_segment_contracts.TypingSegmentStatsPayload,
    ),
    ("TypingSegmentPayload", typing_segment_contracts.TypingSegmentPayload),
    (
        "TypingSegmentIngestResponsePayload",
        typing_segment_contracts.TypingSegmentIngestResponsePayload,
    ),
    (
        "TypingSegmentBatchIngestRequestPayload",
        typing_segment_contracts.TypingSegmentBatchIngestRequestPayload,
    ),
    (
        "TypingSegmentBatchIngestResponsePayload",
        typing_segment_contracts.TypingSegmentBatchIngestResponsePayload,
    ),
    (
        "CapturedTextSourceType",
        captured_text_contracts.CapturedTextSourceType,
    ),
    (
        "CapturedTextSurfaceType",
        captured_text_contracts.CapturedTextSurfaceType,
    ),
    (
        "CapturedTextEventPayload",
        captured_text_contracts.CapturedTextEventPayload,
    ),
    (
        "CapturedTextBatchIngestRequestPayload",
        captured_text_contracts.CapturedTextBatchIngestRequestPayload,
    ),
    (
        "CapturedTextIngestResponsePayload",
        captured_text_contracts.CapturedTextIngestResponsePayload,
    ),
    (
        "CapturedTextBatchIngestResponsePayload",
        captured_text_contracts.CapturedTextBatchIngestResponsePayload,
    ),
    (
        "CapturedTextDebugJobRunRequestPayload",
        captured_text_contracts.CapturedTextDebugJobRunRequestPayload,
    ),
    (
        "CapturedTextDebugJobConfigRequestPayload",
        captured_text_contracts.CapturedTextDebugJobConfigRequestPayload,
    ),
    (
        "CapturedTextDebugJobRunResultPayload",
        captured_text_contracts.CapturedTextDebugJobRunResultPayload,
    ),
    (
        "CapturedTextDebugJobStatusPayload",
        captured_text_contracts.CapturedTextDebugJobStatusPayload,
    ),
)


def build_schema_definitions() -> dict[str, dict[str, Any]]:
    """생성 대상 alias/model의 JSON schema 정의를 모은다."""

    definitions: dict[str, dict[str, Any]] = {}
    for type_name, source in TYPE_SOURCES:
        if isinstance(source, type) and issubclass(source, BaseModel):
            schema = source.model_json_schema(mode="serialization")
        else:
            schema = TypeAdapter(source).json_schema(mode="serialization")
        nested_definitions = schema.pop("$defs", {})
        for nested_name, nested_schema in nested_definitions.items():
            definitions.setdefault(nested_name, nested_schema)
        definitions[type_name] = schema
    return definitions


def render_family_extension_types() -> str:
    """family_extension가 읽는 generated TS type file을 렌더링한다."""

    definitions = build_schema_definitions()
    lines = [
        "// Generated by scripts/codegen/generate_family_extension_types.py.",
        "// Do not edit this file manually.",
        "",
    ]
    for type_name, _source in TYPE_SOURCES:
        lines.extend(render_declaration(type_name, definitions[type_name], definitions))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_family_extension_types(output_path: Path = OUTPUT_PATH) -> None:
    """generated contract 파일을 기록한다."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_family_extension_types(), encoding="utf-8")


def render_declaration(
    type_name: str,
    schema: Mapping[str, Any],
    definitions: Mapping[str, dict[str, Any]],
) -> list[str]:
    """단일 선언을 TS 코드 줄 목록으로 변환한다."""

    if is_interface_schema(schema):
        return render_interface_declaration(type_name, schema, definitions)
    rendered_type = render_schema(schema, definitions)
    return [f"export type {type_name} = {rendered_type};"]


def render_interface_declaration(
    type_name: str,
    schema: Mapping[str, Any],
    definitions: Mapping[str, dict[str, Any]],
) -> list[str]:
    """object schema를 TS interface로 렌더링한다."""

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return [f"export interface {type_name} {{}}"]

    lines = [f"export interface {type_name} {{"]
    for field_name, field_schema in properties.items():
        rendered_field_type = render_schema(field_schema, definitions)
        lines.append(f"  {field_name}: {rendered_field_type};")
    lines.append("}")
    return lines


def render_schema(
    schema: Mapping[str, Any],
    definitions: Mapping[str, dict[str, Any]],
) -> str:
    """JSON schema 조각을 TS type 문자열로 변환한다."""

    ref = schema.get("$ref")
    if isinstance(ref, str):
        return ref.rsplit("/", 1)[-1]

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        return " | ".join(
            dedupe_preserve_order(
                render_schema(member, definitions) for member in any_of
            )
        )

    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        return " | ".join(
            dedupe_preserve_order(
                render_schema(member, definitions) for member in one_of
            )
        )

    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        return " | ".join(json.dumps(value) for value in enum_values)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return " | ".join(
            dedupe_preserve_order(
                render_schema({"type": member_type}, definitions)
                for member_type in schema_type
            )
        )
    if schema_type == "string":
        return "string"
    if schema_type in {"integer", "number"}:
        return "number"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "null":
        return "null"
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, Mapping):
            return f"{render_schema(items, definitions)}[]"
        return "unknown[]"
    if schema_type == "object":
        additional_properties = schema.get("additionalProperties")
        properties = schema.get("properties")
        if isinstance(properties, Mapping) and properties:
            inline_fields = []
            for field_name, field_schema in properties.items():
                inline_fields.append(
                    f"{field_name}: {render_schema(field_schema, definitions)}"
                )
            return "{ " + "; ".join(inline_fields) + " }"
        if additional_properties in (True, None):
            return "Record<string, unknown>"
        if additional_properties is False:
            return "Record<string, never>"
        if isinstance(additional_properties, Mapping):
            return (
                f"Record<string, {render_schema(additional_properties, definitions)}>"
            )
    title = schema.get("title")
    if isinstance(title, str) and title in definitions:
        return title
    return "unknown"


def is_interface_schema(schema: Mapping[str, Any]) -> bool:
    """root schema가 interface로 렌더링될 object인지 판정한다."""

    if schema.get("type") != "object":
        return False
    return isinstance(schema.get("properties"), Mapping)


def dedupe_preserve_order(values: Any) -> list[str]:
    """타입 union 렌더링 시 순서를 유지한 중복 제거."""

    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


if __name__ == "__main__":
    write_family_extension_types()
