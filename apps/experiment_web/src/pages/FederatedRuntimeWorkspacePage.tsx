import { WorkspaceTrackPage } from "./WorkspaceTrackPage";
import type { ExperimentWorkspaceController } from "../hooks/useExperimentWorkspaceController";

export function FederatedRuntimeWorkspacePage(props: {
  controller: ExperimentWorkspaceController;
}) {
  return <WorkspaceTrackPage controller={props.controller} />;
}
