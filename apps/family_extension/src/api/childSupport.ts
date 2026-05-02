import type {
  ChildSupportConversationRequestPayload,
  ChildSupportConversationResponsePayload,
  ChildSupportProactivePromptPayload,
} from "../contracts/generated";
import { requestAgentJson } from "./client";

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
