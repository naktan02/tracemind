import type {
  ChildSupportConversationRequestPayload,
  ChildSupportConversationResponsePayload,
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
