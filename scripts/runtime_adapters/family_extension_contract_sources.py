"""Family extension contract source bridge for scripts codegen."""

from __future__ import annotations

from typing import Any

import agent.src.contracts.captured_text_contracts as captured_text_contracts
import agent.src.contracts.child_support_contracts as child_support_contracts
import agent.src.contracts.family_access_contracts as family_access_contracts
import agent.src.contracts.typing_segment_contracts as typing_segment_contracts
import agent.src.contracts.wellbeing_signal_contracts as wellbeing_signal_contracts

TYPE_SOURCES: tuple[tuple[str, Any], ...] = (
    ("FamilyAccessMode", family_access_contracts.FamilyAccessMode),
    ("FamilyAccessRole", family_access_contracts.FamilyAccessRole),
    ("FamilySetupStatusPayload", family_access_contracts.FamilySetupStatusPayload),
    ("FamilySetupRequestPayload", family_access_contracts.FamilySetupRequestPayload),
    ("FamilySetupResponsePayload", family_access_contracts.FamilySetupResponsePayload),
    ("FamilyUnlockRequestPayload", family_access_contracts.FamilyUnlockRequestPayload),
    (
        "FamilyUnlockResponsePayload",
        family_access_contracts.FamilyUnlockResponsePayload,
    ),
    ("ChildSupportAssistantMode", child_support_contracts.ChildSupportAssistantMode),
    ("ChildSupportSafetyLevel", child_support_contracts.ChildSupportSafetyLevel),
    ("ChildSupportScopeStatus", child_support_contracts.ChildSupportScopeStatus),
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
    ("WellbeingSignalConfidence", wellbeing_signal_contracts.WellbeingSignalConfidence),
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
    ("TypingSegmentSourceType", typing_segment_contracts.TypingSegmentSourceType),
    ("TypingSurfaceType", typing_segment_contracts.TypingSurfaceType),
    ("TypingCaptureConfidence", typing_segment_contracts.TypingCaptureConfidence),
    ("TypingSegmentStatsPayload", typing_segment_contracts.TypingSegmentStatsPayload),
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
    ("CapturedTextSourceType", captured_text_contracts.CapturedTextSourceType),
    ("CapturedTextSurfaceType", captured_text_contracts.CapturedTextSurfaceType),
    ("CapturedTextEventPayload", captured_text_contracts.CapturedTextEventPayload),
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
