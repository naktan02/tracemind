import { RunHistoryPanel } from "../components/RunHistoryPanel";
import { WorkspaceCompareBoard } from "../components/WorkspaceCompareBoard";
import type { ExperimentWorkspaceController } from "../hooks/useExperimentWorkspaceController";
import { formatTrackName } from "../lib/formatters";

export function WorkspaceRecordsPage(props: {
  controller: ExperimentWorkspaceController;
}) {
  const { controller } = props;

  if (!controller.activeTrack) {
    return null;
  }

  const savedWorkspaces = controller.savedWorkspaces.filter(
    (workspace) => workspace.track_name === controller.activeTrack?.track_name,
  );
  const runs = controller.runs.filter(
    (run) => run.track_name === controller.activeTrack?.track_name,
  );

  return (
    <main className="records-page">
      <section className="panel records-hero">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">기록 보기</p>
            <h2>{formatTrackName(controller.activeTrack.track_name)} 기록</h2>
          </div>
        </div>
        <p className="step-intro">
          이 화면은 저장된 실험과 실행 결과만 보는 곳입니다. 같은 기본 조합에서
          방법론만 다르게 한 결과를 먼저 보고, 필요한 경우 그대로 재실행하거나
          복제 후 수정해서 새 실험을 만듭니다.
        </p>
      </section>

      <section className="panel compare-panel compare-panel--top">
        <WorkspaceCompareBoard
          currentWorkspaceId={controller.currentWorkspaceId}
          savedWorkspaces={savedWorkspaces}
          runs={runs}
          rerunningWorkspaceId={controller.rerunningWorkspaceId}
          deletingWorkspaceId={controller.deletingWorkspaceId}
          onRefreshBoard={() => {
            void controller.refreshSavedWorkspaces();
            void controller.refreshRuns();
          }}
          onLoadWorkspace={(workspaceId) =>
            void controller.handleLoadSavedWorkspace(workspaceId)
          }
          onCloneWorkspace={(workspaceId) =>
            void controller.handleCloneWorkspace(workspaceId)
          }
          onRelaunchWorkspace={(workspaceId) =>
            void controller.handleRelaunchWorkspace(workspaceId)
          }
          onDeleteWorkspace={(workspaceId) =>
            void controller.handleDeleteWorkspace(workspaceId)
          }
        />
      </section>

      <section className="panel">
        <RunHistoryPanel
          apiBaseUrl={controller.apiBaseUrl}
          runs={runs}
          runsError={controller.runsError}
          isRunsLoading={controller.isRunsLoading}
          onRefresh={() => void controller.refreshRuns()}
        />
      </section>
    </main>
  );
}
