"""`.env` 로더 테스트."""

from __future__ import annotations

from agent.src.config.env_file import load_agent_env_files, load_env_file


def test_load_env_file_populates_missing_values(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "TRACEMIND_CHILD_SUPPORT_LLM_PROVIDER=ollama",
                "QUOTED_VALUE='hello world'",
            ]
        ),
        encoding="utf-8",
    )
    environ: dict[str, str] = {}

    load_env_file(env_file, environ=environ)

    assert environ["TRACEMIND_CHILD_SUPPORT_LLM_PROVIDER"] == "ollama"
    assert environ["QUOTED_VALUE"] == "hello world"


def test_load_env_file_does_not_override_existing_values(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TRACEMIND_CHILD_SUPPORT_OLLAMA_MODEL=exaone3.5:2.4b",
        encoding="utf-8",
    )
    environ = {"TRACEMIND_CHILD_SUPPORT_OLLAMA_MODEL": "qwen3:8b"}

    load_env_file(env_file, environ=environ)

    assert environ["TRACEMIND_CHILD_SUPPORT_OLLAMA_MODEL"] == "qwen3:8b"


def test_load_agent_env_files_prefers_agent_local_values(
    tmp_path,
    monkeypatch,
) -> None:
    repo_root = tmp_path
    agent_dir = repo_root / "agent"
    agent_dir.mkdir()
    (repo_root / ".env").write_text(
        "TRACEMIND_CHILD_SUPPORT_OLLAMA_MODEL=qwen3:8b",
        encoding="utf-8",
    )
    (agent_dir / ".env").write_text(
        "TRACEMIND_CHILD_SUPPORT_OLLAMA_MODEL=exaone3.5:2.4b",
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.src.config.env_file.AGENT_ENV_PATH", agent_dir / ".env")
    monkeypatch.setattr("agent.src.config.env_file.ROOT_ENV_PATH", repo_root / ".env")
    environ: dict[str, str] = {}

    load_agent_env_files(environ=environ)

    assert environ["TRACEMIND_CHILD_SUPPORT_OLLAMA_MODEL"] == "exaone3.5:2.4b"
