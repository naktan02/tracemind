import type { ChildSupportProactivePromptPayload } from "../../contracts/generated";
import { ChildSupportChatSurface, type CoachMessage } from "./ChildSupportChatSurface";

type ProactiveCoachPopupProps = {
  prompt: ChildSupportProactivePromptPayload;
  onDismiss: () => void;
};

export function ProactiveCoachPopup({
  prompt,
  onDismiss,
}: ProactiveCoachPopupProps) {
  const promptText = prompt.prompt_text;
  if (promptText == null) {
    return null;
  }
  const initialMessage: CoachMessage = {
    id: `proactive-popup-${promptText}`,
    role: "assistant",
    text: promptText,
    createdAt: new Date().toISOString(),
  };

  return (
    <aside
      className="proactive-coach-popup"
      aria-label="AI 마음 도움 선제 대화"
    >
      <div className="proactive-coach-header">
        <div>
          <p className="coach-eyebrow">AI 마음 도움</p>
          <h3>잠깐 같이 확인해요</h3>
        </div>
        <button
          className="proactive-coach-close"
          type="button"
          onClick={onDismiss}
          aria-label="AI 마음 도움 팝업 닫기"
        >
          x
        </button>
      </div>
      <ChildSupportChatSurface
        initialMessage={initialMessage}
        initialSuggestions={prompt.suggested_prompts}
        compact
      />
    </aside>
  );
}
