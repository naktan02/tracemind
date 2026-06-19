"""아이용 지원 대화 LLM provider adapter."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol

from agent.src.contracts.child_support_contracts import (
    ChildSupportAssistantMode,
)

CHILD_SUPPORT_LLM_PROVIDER_ENV = "TRACEMIND_CHILD_SUPPORT_LLM_PROVIDER"
CHILD_SUPPORT_OLLAMA_BASE_URL_ENV = "TRACEMIND_CHILD_SUPPORT_OLLAMA_BASE_URL"
CHILD_SUPPORT_OLLAMA_MODEL_ENV = "TRACEMIND_CHILD_SUPPORT_OLLAMA_MODEL"

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

ChildSupportLlmRole = Literal["system", "user", "assistant"]


class ChildSupportLlmError(RuntimeError):
    """LLM provider 호출 실패."""


@dataclass(frozen=True, slots=True)
class ChildSupportLlmMessage:
    """provider에 전달할 구조화된 대화 메시지."""

    role: ChildSupportLlmRole
    content: str

    def to_ollama_message(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }


class ChildSupportLlmProvider(Protocol):
    """child-support 응답 생성을 위한 LLM adapter protocol."""

    @property
    def assistant_mode(self) -> ChildSupportAssistantMode:
        """응답 payload에 기록할 assistant mode."""
        ...

    def generate_reply(
        self,
        *,
        prompt: str,
        messages: tuple[ChildSupportLlmMessage, ...] = (),
    ) -> str:
        """prompt 또는 구조화 message를 받아 아이에게 보여줄 응답을 생성한다."""
        ...


@dataclass(frozen=True, slots=True)
class OllamaChildSupportLlmProvider:
    """로컬 Ollama HTTP API 기반 child-support LLM provider."""

    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout_seconds: float = 20.0

    @property
    def assistant_mode(self) -> ChildSupportAssistantMode:
        return ChildSupportAssistantMode.LOCAL_LLM

    def generate_reply(
        self,
        *,
        prompt: str,
        messages: tuple[ChildSupportLlmMessage, ...] = (),
    ) -> str:
        ollama_messages = (
            [message.to_ollama_message() for message in messages]
            if messages
            else [{"role": "user", "content": prompt}]
        )
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": 0.35,
                "top_p": 0.9,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw_body = response.read().decode("utf-8")
        except (OSError, urllib.error.HTTPError) as exc:
            raise ChildSupportLlmError(f"Ollama 호출 실패: {exc}") from exc

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ChildSupportLlmError(
                "Ollama 응답을 JSON으로 읽지 못했습니다."
            ) from exc

        reply = _extract_ollama_chat_content(body)
        if not isinstance(reply, str) or not reply.strip():
            raise ChildSupportLlmError("Ollama 응답에 message.content가 없습니다.")
        return reply.strip()


def build_child_support_llm_provider_from_env(
    environ: Mapping[str, str] | None = None,
) -> ChildSupportLlmProvider | None:
    """환경변수에서 child-support LLM provider를 구성한다."""

    effective_environ = os.environ if environ is None else environ
    provider_name = effective_environ.get(CHILD_SUPPORT_LLM_PROVIDER_ENV, "").strip()
    if provider_name in {"", "local_guarded", "none"}:
        return None
    if provider_name != "ollama":
        raise ValueError(
            "TRACEMIND_CHILD_SUPPORT_LLM_PROVIDER는 "
            "'ollama', 'local_guarded', 'none' 중 하나여야 합니다."
        )
    return OllamaChildSupportLlmProvider(
        base_url=effective_environ.get(
            CHILD_SUPPORT_OLLAMA_BASE_URL_ENV,
            DEFAULT_OLLAMA_BASE_URL,
        ).strip()
        or DEFAULT_OLLAMA_BASE_URL,
        model=effective_environ.get(
            CHILD_SUPPORT_OLLAMA_MODEL_ENV,
            DEFAULT_OLLAMA_MODEL,
        ).strip()
        or DEFAULT_OLLAMA_MODEL,
    )


def _extract_ollama_chat_content(body: object) -> str | None:
    if not isinstance(body, dict):
        return None
    message = body.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None
