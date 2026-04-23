export interface WorkspaceStepItem {
  stepId: string;
  label: string;
  detail: string;
  tone: "complete" | "active" | "pending";
}

export function WorkspaceStepRail(props: {
  activeStepId: string;
  steps: WorkspaceStepItem[];
  onStepChange: (stepId: string) => void;
}) {
  return (
    <div className="workflow-steps">
      {props.steps.map((step, index) => (
        <button
          key={step.stepId}
          type="button"
          className={
            step.stepId === props.activeStepId
              ? "workflow-step workflow-step--active"
              : `workflow-step workflow-step--${step.tone}`
          }
          onClick={() => props.onStepChange(step.stepId)}
        >
          <span className="workflow-step__index">{index + 1}</span>
          <span className="workflow-step__body">
            <strong>{step.label}</strong>
            <small>{step.detail}</small>
          </span>
        </button>
      ))}
    </div>
  );
}
