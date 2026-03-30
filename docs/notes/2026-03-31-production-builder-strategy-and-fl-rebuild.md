# 2026-03-31 Production Builder Strategy And FL Rebuild

## User

우회말고 왜 안되는거지? 아까 `k=2` 이런거 하는 곳에서 잘 사용하도록 만들지 않았나? 그걸 공통으로 각 스크립트 파일에서 사용 안하나? 제대로 먹도록 리펙터링 진행하면서 다른 파일들도 확인해봐.

- `single`을 `multi`의 특수 케이스로 올리는 runtime/prototype contract 리팩터링도 진행해

## Assistant

Hydra preset override 문제는 사용법이 아니라 config group 위치 문제라고 설명했다. `run_federated_simulation`의 `federated_run_preset`이 실험 하위 상대 group로 잡혀 있어 override 시 dict 대신 문자열로 치환되고 있었고, 이를 top-level group으로 올려 `federated_run_preset=standard federated_run_preset.rounds=3` 형태가 그대로 동작하도록 고치겠다고 답했다.

또 runtime/prototype contract는 `single = 길이 1 리스트`로 올리는 방향이 맞다고 설명하고, consumer 쪽을 먼저 multi-friendly로 바꾸겠다고 정리했다.

## User

이제 언제든지 멀티로 가능하다는건가? 그리고 어떤 구조로 바꾼건데? 그래서 싱글하고 멀티랑 어떤게 교환가능한데?

## Assistant

아직 end-to-end는 아니고, runtime/contract는 멀티를 받을 수 있게 되었지만 기본 생성 경로는 아직 single이라고 설명했다.

- `PrototypePackPayload.categories`를 `category -> prototypes[]`로 바꿈
- `single`은 길이 1 리스트로 표현
- scoring은 category 내부 여러 prototype 중 최고 점수를 선택
- runtime consumer는 single/multi pack 둘 다 읽을 수 있음
- producer/build 경로는 아직 single 생성기만 사용 중

## User

그럼 이제 그 생성 부분 진행해야 하는거지?

## Assistant

네. end-to-end 멀티 prototype을 원하면 생성 부분이 다음 단계라고 설명했다.

- 현재 runtime/scoring은 멀티를 소비할 준비가 끝남
- 남은 건 `seed_prototypes`, FL 재빌드, dataset pipeline이 어떤 builder 전략으로 prototype을 만들지 결정하는 부분
- 실험용 전략 모듈을 그대로 끌어다 붙이지 말고 production용 strategy 경계를 따로 두는 게 맞다고 설명

## User

production builder를 strategy화, FL 재빌드 경로에서 그 builder strategy를 쓰게 만들기 이 두개 작업 진행하는거지? 진행해

## Assistant

작업 계획:

1. 현재 prototype 생성 경로와 FL 재빌드 연결 지점 파악
2. production builder strategy 추상화와 `single/kmeans` 구현 추가
3. `seed_prototypes` 및 FL 재빌드 경로를 strategy 기반으로 연결
4. Hydra 설정, 테스트, README 정리 및 검증

## Assistant Work Log

확인한 핵심 파일:

- `scripts/prototypes/prototype_pack_builder.py`
- `scripts/prototypes/seed_prototypes.py`
- `scripts/experiments/run_federated_simulation.py`
- `scripts/experiments/prototype_strategy/strategies.py`
- `scripts/datasets/run_dataset_pipeline.py`
- `shared/src/contracts/prototype_build_state_contracts.py`
- `shared/src/contracts/prototype_contracts.py`

핵심 판단:

- `PrototypePackBuilder`는 exact incremental build-state를 지원하는 single-centroid builder로 유지
- production 경로에는 그 위에 `prototype build strategy` 계층을 추가
- `single` 전략은 기존 builder를 감싸고 build-state까지 생성
- `kmeans` 전략은 multi-prototype pack만 생성하고 build-state는 생성하지 않음
- FL 재빌드와 seed, dataset pipeline이 같은 strategy group을 타도록 연결

## Assistant

구현 후 설명:

- 새 production 전략 모듈 `scripts/prototypes/build_strategies.py` 추가
  - `PrototypeBuildRequest`
  - `PrototypeBuildArtifacts`
  - `PrototypeBuildStrategy`
  - `SinglePrototypeBuildStrategy`
  - `KMeansPrototypeBuildStrategy`
- 새 Hydra group `scripts/conf/prototype_builder/`
  - `single.yaml`
  - `kmeans.yaml`
- `scripts/conf/prototypes/seed_prototypes.yaml`
  - `prototype_builder=single` 기본값 추가
- `scripts/conf/experiments/run_federated_simulation.yaml`
  - `prototype_builder=single` 기본값 추가
- `scripts/conf/datasets/run_dataset_pipeline.yaml`
  - `prototype_builder=single` 기본값 추가
- `scripts/prototypes/seed_prototypes.py`
  - builder strategy를 instantiate해서 사용
  - kmeans 선택 시 pack은 생성하지만 build-state는 `None`
  - manifest에 `prototype_builder` 정보 기록
- `scripts/datasets/run_dataset_pipeline.py`
  - dataset pipeline prototype stage도 같은 strategy를 사용
- `scripts/experiments/run_federated_simulation.py`
  - bootstrap 기반 prototype 재빌드가 `prototype_build_strategy`를 사용
- `shared/src/contracts/README.md`
  - build-state v1이 single exact incremental 전용임을 명시
- `scripts/prototypes/update_prototype_build_state.py`
  - single-centroid build-state용 스크립트라는 점을 명확히 표시

## Verification

Lint:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  scripts/prototypes/build_strategies.py \
  scripts/prototypes/seed_prototypes.py \
  scripts/prototypes/prototype_pack_builder.py \
  scripts/prototypes/update_prototype_build_state.py \
  scripts/datasets/run_dataset_pipeline.py \
  scripts/experiments/run_federated_simulation.py \
  shared/src/contracts/prototype_build_state_contracts.py \
  tests/unit/test_prototype_build_strategies.py \
  tests/unit/test_run_federated_simulation.py \
  tests/unit/test_scripts_hydra_configs.py
```

결과: 통과

Pytest:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/unit/test_prototype_build_strategies.py \
  tests/unit/test_scripts_hydra_configs.py \
  tests/unit/test_run_federated_simulation.py \
  shared/tests/unit/test_prototype_contracts.py \
  agent/tests/unit/test_scoring_service.py \
  agent/tests/unit/test_prototype_runtime_and_sync.py
```

결과: `24 passed`

실제 smoke 실행:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/experiments/run_federated_simulation.py \
  embedding=hash_debug \
  runtime=cpu_local \
  federated_run_preset=smoke \
  prototype_builder=kmeans \
  prototype_builder.candidate_ks=[2]
```

출력:

- `initial_model_revision=sim_rev_0000`
- `initial_prototype_version=proto_sim_0000`
- `final_model_revision=sim_rev_0001`
- `final_prototype_version=proto_sim_0001`
- `round_count=1`

생성된 pack 확인:

- `tmp/federated_simulation_smoke/main_server/prototype_packs/proto_sim_0001.json`
- category별 prototype 개수:
  - `anxiety: 2`
  - `depression: 2`
  - `normal: 2`
  - `suicidal: 2`

## Current Status

- runtime/scoring: single/multi 모두 소비 가능
- production seed: strategy 기반으로 single/kmeans 선택 가능
- FL 재빌드: strategy 기반으로 single/kmeans 선택 가능
- exact incremental build-state: 아직 single mean-centroid 전용

## Remaining Follow-up

다음 확장 후보:

1. production builder metadata를 manifest 외 payload 레벨까지 올릴지 결정
2. FL runtime에서 `max cosine` 대신 `log-sum-exp` 같은 multi scoring 정책이 필요한지 판단
3. build-state를 multi-prototype까지 일반화할지, 아니면 single exact / multi rebuild-only로 유지할지 결정

---

## Additional Follow-up: Selection Diagnostics

### User

보이게 만들면서 scoring/prototype 분리력이 왜 필요한거야?

1. 보이게 만들아
2. scoring/prototype 분리력이 왜 필요한거야?

### Assistant

현재 `normal`이 threshold에서 죽는지, cap에서 잘리는지 구분하려면 selector가 두 단계를 metadata로 남겨야 한다고 설명했다.

- `threshold_accepted`
- `selected_by_cap`
- `selection_stage`
- 필요하면 `pre_cap_rank`

이 값을 round/client별 JSONL과 summary로 떨구면 원인 분리가 된다고 정리했다.

### Assistant Work Log

수정 사항:

- `agent/src/services/training/pseudo_label_service.py`
  - selector metadata에
    - `threshold_accepted`
    - `selected_by_cap`
    - `final_accepted`
    - `selection_stage`
    - `max_examples`
    - `pre_cap_rank`
    추가
- `scripts/experiments/run_federated_simulation.py`
  - `save_selection_diagnostics(...)` 추가
  - 각 client/round마다
    - `selection_dumps/<round>.candidates.jsonl`
    - `selection_dumps/<round>.summary.json`
    저장
- `agent/tests/unit/test_local_training_service.py`
  - `accepted / dropped_by_cap / threshold_rejected`가 구분되는지 검증
- `tests/unit/test_run_federated_simulation.py`
  - simulation이 dump 파일을 실제로 생성하는지 검증
- `scripts/README.md`
  - `selection_dumps/` 산출물 설명 추가

### Verification

Lint:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  agent/src/services/training/pseudo_label_service.py \
  agent/tests/unit/test_local_training_service.py \
  scripts/experiments/run_federated_simulation.py \
  tests/unit/test_run_federated_simulation.py \
  scripts/README.md
```

결과: 통과

Pytest:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  agent/tests/unit/test_local_training_service.py \
  tests/unit/test_run_federated_simulation.py \
  tests/unit/test_prototype_build_strategies.py \
  tests/unit/test_scripts_hydra_configs.py
```

결과: `15 passed`

실제 smoke 실행:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/experiments/run_federated_simulation.py \
  embedding=hash_debug \
  runtime=cpu_local \
  federated_run_preset=smoke
```

생성된 파일:

- `tmp/federated_simulation_smoke/agents/agent_01/selection_dumps/round_0001.candidates.jsonl`
- `tmp/federated_simulation_smoke/agents/agent_01/selection_dumps/round_0001.summary.json`
- `tmp/federated_simulation_smoke/agents/agent_02/selection_dumps/round_0001.candidates.jsonl`
- `tmp/federated_simulation_smoke/agents/agent_02/selection_dumps/round_0001.summary.json`
- `tmp/federated_simulation_smoke/agents/agent_03/selection_dumps/round_0001.candidates.jsonl`
- `tmp/federated_simulation_smoke/agents/agent_03/selection_dumps/round_0001.summary.json`
- `tmp/federated_simulation_smoke/agents/agent_04/selection_dumps/round_0001.candidates.jsonl`
- `tmp/federated_simulation_smoke/agents/agent_04/selection_dumps/round_0001.summary.json`

예시:

- `agent_03` summary:
  - `accepted: 32`
  - `dropped_by_cap: 1670`
  - `threshold_rejected: 10840`
  - true `normal` 중 `accepted: 28`, `dropped_by_cap: 1328`, `threshold_rejected: 8575`
- candidate JSONL에는 row별로
  - `confidence`
  - `margin`
  - `threshold_accepted`
  - `selected_by_cap`
  - `selection_stage`
  - `pre_cap_rank`
  - `category_scores`
  가 들어간다.
