import { formatDateTime } from "../lib/formatters";
import type { SavedWorkspaceSummaryPayload } from "../types";

export function SavedWorkspacePanel(props: {
  currentWorkspaceId: string | null;
  savedWorkspaces: SavedWorkspaceSummaryPayload[];
  savedWorkspacesError: string | null;
  isSavedWorkspacesLoading: boolean;
  loadingWorkspaceId: string | null;
  onRefresh: () => void;
  onLoadWorkspace: (workspaceId: string) => void;
}) {
  return (
    <div className="subpanel">
      <div className="subpanel__header">
        <div>
          <p className="panel-kicker">Saved</p>
          <h3>Workspaces</h3>
        </div>
        <button
          type="button"
          className="ghost-button ghost-button--small"
          onClick={props.onRefresh}
          disabled={props.isSavedWorkspacesLoading}
        >
          {props.isSavedWorkspacesLoading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {props.savedWorkspacesError ? (
        <div className="message-block message-block--error">
          <h3>Workspace list error</h3>
          <p>{props.savedWorkspacesError}</p>
        </div>
      ) : null}

      <div className="saved-workspace-list">
        {props.savedWorkspaces.length > 0 ? (
          props.savedWorkspaces.map((workspace) => (
            <article
              className={
                workspace.workspace_id === props.currentWorkspaceId
                  ? "saved-workspace-card saved-workspace-card--active"
                  : "saved-workspace-card"
              }
              key={workspace.workspace_id}
            >
              <div className="saved-workspace-card__body">
                <strong>{workspace.workspace_id}</strong>
                <span>
                  {workspace.track_name} / {workspace.entrypoint_name}
                </span>
                <span>manifest: {workspace.manifest_id}</span>
                <span>updated: {formatDateTime(workspace.updated_at)}</span>
                {workspace.latest_run_id ? (
                  <code>latest run: {workspace.latest_run_id}</code>
                ) : null}
              </div>
              <button
                type="button"
                className="ghost-button ghost-button--small"
                onClick={() => props.onLoadWorkspace(workspace.workspace_id)}
                disabled={props.loadingWorkspaceId === workspace.workspace_id}
              >
                {props.loadingWorkspaceId === workspace.workspace_id
                  ? "Loading..."
                  : "Load"}
              </button>
            </article>
          ))
        ) : (
          <p className="hint-text">아직 저장된 workspace가 없습니다.</p>
        )}
      </div>
    </div>
  );
}
