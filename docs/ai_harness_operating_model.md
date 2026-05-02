# TraceMind AI Harness Operating Model

이 문서는 하네스 자체를 유지보수할 때만 읽는 maintainer 문서다.
평소 작업 진입점은 `AGENTS.md`, path-specific `AGENTS.md`,
`docs/ai_context_manifest.yaml`을 우선한다.

## canonical layers

1. `.codex/config.toml`
2. root `AGENTS.md`
3. path-specific `AGENTS.md`
4. `docs/ai_context_manifest.yaml`
5. `.codex/skills/*`

## maintainer-only documents

- `docs/ai_harness_operating_model.md`
- `docs/ai_harness_eval_cases.yaml`
- `.agents/rules/design.md`는 legacy compatibility only

## execution permissions

- Project default: `approval_policy=on-request`,
  `sandbox_mode=danger-full-access`.
- GPU preflight, tests, lint, local smoke, and repo-scoped experiment smoke should run
  in the real execution environment when they are directly needed for the task.
- Destructive commands, repo-outside writes, credential changes, commit/push, large
  downloads, paid/external data transfer, and unrelated process termination still
  require explicit user confirmation.
- Operational details live in `docs/operations/local-runbook.md`.

## skill usage

- TraceMind planning, experiment intent alignment, or skill routing:
  `tracemind-research-loop`
- ambiguous experiment terms, algorithm comparison, or plan stress test:
  `grill-with-docs`
- contract shape or field meaning change: `tracemind-contract-sync`
- adapter family or backend addition: `tracemind-strategy-addition`
- FL round mismatch or artifact drift debug: `tracemind-federation-debug`
- bug, failure, regression, or performance issue: `diagnose`
- test-first feature or contract behavior change: `tdd`
- unfamiliar code area map: `zoom-out`
- deep module or deletion-test architecture review:
  `improve-codebase-architecture`
- active docs drift cleanup: `tracemind-doc-sync`

## upstream skills

- Matt Pocock `skills` 중 필요한 workflow는 project-local Codex skill로 일부를
  원형 보존해 사용한다.
- 출처와 license는 `.codex/skills/THIRD_PARTY_NOTICES.md`에 둔다.
- 원본 workflow를 active docs에 다시 복제하지 않는다. TraceMind 특화 라우팅은
  `.codex/skills/tracemind-research-loop/SKILL.md`가 소유한다.

## maintenance rule

- 자동으로 읽히는 instruction layer는 짧게 유지한다.
- 반복되는 규칙은 AGENTS 또는 skill로 올리고, 설명 문서는 보조 계층으로 둔다.
- source of truth는 코드 가까이에 두고, notes는 archive로 유지한다.
