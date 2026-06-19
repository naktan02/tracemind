import type {
  ChildSupportConversationRequestPayload,
  ChildSupportConversationResponsePayload,
  ChildSupportProactivePromptClaimRequestPayload,
  ChildSupportProactivePromptPayload,
} from "../../contracts/generated";
import { requestAgentJson } from "../../common/agentClient";

export async function createChildSupportMessage(
  payload: ChildSupportConversationRequestPayload,
): Promise<ChildSupportConversationResponsePayload> {
  return requestAgentJson("/api/v1/child-support/messages", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getChildSupportProactivePrompt(): Promise<ChildSupportProactivePromptPayload> {
  return requestAgentJson("/api/v1/child-support/proactive-prompt");
}

export async function claimChildSupportProactivePrompt(
  payload: ChildSupportProactivePromptClaimRequestPayload,
): Promise<ChildSupportProactivePromptPayload> {
  return requestAgentJson("/api/v1/child-support/proactive-prompt/claim", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
