import { useEffect, useRef, useState } from "react";

import { createChildSupportMessage } from "../api/childSupport";
import type {
  ChildSupportConversationResponsePayload,
  ChildSupportSuggestionPayload,
} from "../../contracts/generated";

export type CoachMessage = {
  id: string;
  role: "assistant" | "child";
  text: string;
  createdAt: string;
  response?: ChildSupportConversationResponsePayload;
};

export type AssistantPromptSeed = {
  id: string;
  text: string;
  suggestions?: ChildSupportSuggestionPayload[];
};

export const STARTER_SUGGESTIONS: ChildSupportSuggestionPayload[] = [
  {
    id: "starter-breathe",
    label: "숨 쉬기 도와줘",
    prompt: "지금 30초 동안 따라 할 수 있는 숨 쉬기 방법을 알려줘.",
  },
  {
    id: "starter-feeling",
    label: "감정 고르기",
    prompt: "내가 느끼는 감정을 고를 수 있게 쉬운 보기로 알려줘.",
  },
  {
    id: "starter-parent",
    label: "부모님께 말하기",
    prompt: "부모님께 보여줄 수 있게 지금 마음을 짧게 정리해줘.",
  },
];

type ChildSupportChatSurfaceProps = {
  initialMessage: CoachMessage;
  initialSuggestions?: ChildSupportSuggestionPayload[];
  promptSeed?: AssistantPromptSeed | null;
  compact?: boolean;
  onAssistantModeChange?: (assistantMode: string) => void;
};

function createLocalMessage(
  role: CoachMessage["role"],
  text: string,
  response?: ChildSupportConversationResponsePayload,
): CoachMessage {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    text,
    createdAt: new Date().toISOString(),
    response,
  };
}

function resolveSafetyLabel(
  response: ChildSupportConversationResponsePayload,
): string {
  if (response.safety_level === "urgent") {
    return "지금 바로 어른에게 알리기";
  }
  if (response.safety_level === "parent_handoff") {
    return "어른과 함께 확인하기";
  }
  if (response.safety_level === "check_in") {
    return "천천히 확인하기";
  }
  return "대화 이어가기";
}

export function ChildSupportChatSurface({
  initialMessage,
  initialSuggestions = STARTER_SUGGESTIONS,
  promptSeed,
  compact = false,
  onAssistantModeChange,
}: ChildSupportChatSurfaceProps) {
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<CoachMessage[]>([initialMessage]);
  const [suggestions, setSuggestions] =
    useState<ChildSupportSuggestionPayload[]>(initialSuggestions);
  const [isSending, setIsSending] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const consumedSeedIdsRef = useRef<Set<string>>(new Set());
  const threadEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    threadEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages.length, isSending]);

  useEffect(() => {
    if (
      promptSeed == null ||
      consumedSeedIdsRef.current.has(promptSeed.id) ||
      !promptSeed.text.trim()
    ) {
      return;
    }
    consumedSeedIdsRef.current.add(promptSeed.id);
    setMessages((current) => [
      ...current,
      createLocalMessage("assistant", promptSeed.text),
    ]);
    if (promptSeed.suggestions && promptSeed.suggestions.length > 0) {
      setSuggestions(promptSeed.suggestions);
    }
  }, [promptSeed]);

  async function submitMessage(seedPrompt?: string) {
    const message = (seedPrompt ?? draft).trim();
    if (!message || isSending) {
      return;
    }

    setErrorMessage(null);
    setDraft("");
    setIsSending(true);
    setMessages((current) => [...current, createLocalMessage("child", message)]);

    try {
      const response = await createChildSupportMessage({
        message,
        conversation_id: conversationId,
      });
      setConversationId(response.conversation_id);
      onAssistantModeChange?.(response.assistant_mode.replace("_", " "));
      setMessages((current) => [
        ...current,
        createLocalMessage("assistant", response.reply_text, response),
      ]);
      if (response.suggested_prompts.length > 0) {
        setSuggestions(response.suggested_prompts);
      }
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "마음 도움 응답을 아직 만들지 못했습니다.";
      setErrorMessage(message);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className={compact ? "coach-chat-surface compact" : "coach-chat-surface"}>
      <div className="coach-thread" aria-live="polite">
        {messages.map((message) => (
          <article
            key={message.id}
            className={
              message.role === "assistant"
                ? "coach-bubble-row assistant"
                : "coach-bubble-row child"
            }
          >
            <div className="coach-bubble">
              <p>{message.text}</p>
              {message.response && (
                <div className="coach-response-meta">
                  <span>{resolveSafetyLabel(message.response)}</span>
                  <span>{message.response.assistant_mode}</span>
                </div>
              )}
              {message.response?.parent_handoff_suggested && (
                <div className="coach-handoff">
                  {message.response.parent_handoff_label}
                </div>
              )}
            </div>
          </article>
        ))}
        {isSending && (
          <article className="coach-bubble-row assistant">
            <div className="coach-bubble coach-bubble-waiting">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
              <p>응답을 조심스럽게 고르는 중...</p>
            </div>
          </article>
        )}
        <div ref={threadEndRef} className="coach-thread-end" aria-hidden="true" />
      </div>

      <div className="coach-suggestions" aria-label="추천 입력">
        {suggestions.map((suggestion) => (
          <button
            key={suggestion.id}
            className="coach-suggestion-chip"
            type="button"
            onClick={() => void submitMessage(suggestion.prompt)}
            disabled={isSending}
          >
            {suggestion.label}
          </button>
        ))}
      </div>

      {errorMessage && (
        <div className="coach-error" role="status">
          {errorMessage}
        </div>
      )}

      <form
        className="coach-composer"
        onSubmit={(event) => {
          event.preventDefault();
          void submitMessage();
        }}
      >
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void submitMessage();
            }
          }}
          rows={compact ? 1 : 2}
          placeholder="예: 오늘 친구 말 때문에 마음이 계속 불편해"
        />
        <button
          className="coach-send-button"
          type="submit"
          disabled={isSending || !draft.trim()}
        >
          보내기
        </button>
      </form>
    </div>
  );
}
