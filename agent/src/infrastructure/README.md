# Agent Infrastructure

`infrastructure/`는 agent runtime이 쓰는 외부/저장소 adapter를 소유한다. runtime
orchestration과 policy 판단은 `agent/src/runtime/` 또는 `agent/src/services/`에 두고,
이 디렉터리는 저장, 모델 adapter, transport 같은 mechanism만 둔다.

하위 경계:

- `repositories/`: agent-local SQLite/file 저장소와 저장 lifecycle.
- `persistence/`: 저장소 공통 persistence primitive 자리.
- `model_adapters/`: embedding/translation adapter와 model-runtime helper.
- `transport/`: HTTP/IPC 같은 외부 통신 adapter 자리.
