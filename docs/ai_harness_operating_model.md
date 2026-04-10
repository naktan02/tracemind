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

## skill usage

- contract shape or field meaning change: `tracemind-contract-sync`
- adapter family or backend addition: `tracemind-strategy-addition`
- FL round mismatch or artifact drift debug: `tracemind-federation-debug`
- active docs drift cleanup: `tracemind-doc-sync`

## maintenance rule

- 자동으로 읽히는 instruction layer는 짧게 유지한다.
- 반복되는 규칙은 AGENTS 또는 skill로 올리고, 설명 문서는 보조 계층으로 둔다.
- source of truth는 코드 가까이에 두고, notes는 archive로 유지한다.
