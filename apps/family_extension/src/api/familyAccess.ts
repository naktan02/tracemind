import type {
  FamilyAccessRole,
  FamilySetupRequestPayload,
  FamilySetupResponsePayload,
  FamilySetupStatusPayload,
  FamilyUnlockResponsePayload,
} from "../contracts/generated";
import { requestAgentJson } from "./client";

export async function fetchFamilySetupStatus(): Promise<FamilySetupStatusPayload> {
  return requestAgentJson("/api/v1/family/setup/status");
}

export async function submitInitialFamilySetup(
  request: FamilySetupRequestPayload,
): Promise<FamilySetupResponsePayload> {
  return requestAgentJson("/api/v1/family/setup", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function unlockFamilyRole(
  role: FamilyAccessRole,
  pin: string,
): Promise<FamilyUnlockResponsePayload> {
  return requestAgentJson("/api/v1/family/unlock", {
    method: "POST",
    body: JSON.stringify({ role, pin }),
  });
}
