import { useEffect, useState } from "react";

import { getChildSupportProactivePrompt } from "../api/childSupport";
import {
  ChildSupportChatSurface,
  type AssistantPromptSeed,
  type CoachMessage,
} from "./ChildSupportChatSurface";

const INITIAL_MESSAGE: CoachMessage = {
  id: "welcome",
  role: "assistant",
  text:
    "요즘 혼자 버티는 시간이 많았던 것 같아요. 지금 가장 힘든 부분을 천천히 말해줄 수 있나요?",
  createdAt: new Date().toISOString(),
};

export function ChildSupportCoachPanel() {
  const [assistantMode, setAssistantMode] = useState("local guarded");
  const [promptSeed, setPromptSeed] = useState<AssistantPromptSeed | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadProactivePrompt() {
      try {
        const prompt = await getChildSupportProactivePrompt();
        const promptText = prompt.prompt_text;
        if (cancelled || !prompt.should_prompt || promptText === null) {
          return;
        }
        setPromptSeed({
          id: `coach-page-${promptText}`,
          text: promptText,
          suggestions: prompt.suggested_prompts,
        });
      } catch {
        // 화면 진입 보조 문구는 실패해도 기본 대화 UI를 막지 않는다.
      }
    }

    void loadProactivePrompt();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="coach-shell" aria-label="본인 페이지 AI 마음 도움">
      <div className="coach-header">
        <div>
          <p className="coach-eyebrow">AI 마음 도움</p>
          <h3>지금 마음을 작게 꺼내놓는 곳</h3>
          <p className="coach-copy">
            최근 상태 흐름을 참고해 대화를 이어가고, 필요할 때 도움 받을 곳을
            함께 안내합니다.
          </p>
        </div>
        <div className="coach-mode-pill">
          <span className="coach-mode-dot" />
          {assistantMode}
        </div>
      </div>

      <ChildSupportChatSurface
        initialMessage={INITIAL_MESSAGE}
        promptSeed={promptSeed}
        onAssistantModeChange={setAssistantMode}
      />

      <p className="coach-disclaimer">
        위험하거나 혼자 감당하기 어려운 상황이면 지금 바로 보호자나 믿을 수
        있는 어른에게 알려 주세요.
      </p>
    </section>
  );
}
