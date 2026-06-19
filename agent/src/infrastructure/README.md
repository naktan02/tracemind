# Agent Infrastructure

`infrastructure/`는 agent runtime이 쓰는 공통 외부/저장소 adapter를 소유한다.
runtime orchestration과 policy 판단은 `agent/src/runtime/` 또는
`agent/src/features/`에 두고, 이 디렉터리는 저장, 모델 adapter, transport 같은
mechanism만 둔다.

하위 경계:

- `repositories/`: 여러 feature가 공유하는 agent-local SQLite/file 저장소와 저장
  lifecycle. feature 전용 storage는 해당 `features/<feature>/storage/`에 둔다.
- `model_adapters/`: embedding/translation adapter와 model-runtime helper.
