# Scripts AGENTS

## 역할

`scripts/`는 runtime 코어를 조합하는 실험층이다. thin wrapper, report,
sweep, visualization, exploratory-only logic만 둔다.

## 먼저 읽을 것

1. `scripts/README.md`
2. 논문 트랙이면 `docs/contracts/central_peft_text_encoder_trainer_contract.md`
3. `conf/`의 관련 Hydra config group
4. 작업 대상 entrypoint의 README 또는 모듈

## 변경 규칙

- 운영 후보 알고리즘을 `scripts`에 먼저 만들고 나중에 복사하지 않는다.
- production/runtime에서 재사용되어야 하는 알고리즘 코어는 `methods`로 올린다.
  공통 contract/domain은 `shared`, runtime adapter는 `agent`/`main_server`에 둔다.
- 실행 설정의 source of truth는 `conf/` Hydra config group이다.
- `scripts`는 method, trainable surface, teacher 구현 이름으로 분기하지
  않는다. entrypoint는 `conf`가 고른 축을 canonical request로 넘기고, 구현 의미는
  `methods`의 descriptor/hook/runner contract가 소유한다.
- `scripts/support/**/teacher_providers` 같은 provider family 폴더를 새로 늘리지
  않는다. teacher source는 method/recipe 내부 hook이고, scripts는 필요한 경우 source를 호출하는
  orchestration/IO adapter까지만 둔다.
- `execution_context/dataset_asset`, `execution_context/embedding_adapter`,
  `execution_context/runtime_env` selector를 기본 실행 축으로 본다.
- 스크립트용 preset을 Python helper 파일에 다시 복제하지 않는다.
- 일반 dataset/prototype/cache warm-up 스크립트의 기본 runtime은 `gpu_online`으로
  간주한다. FL SSL smoke/main/sweep은 `gpu_local + mxbai`를 기본 논문용 실행
  기준으로 본다.

## 테스트 규칙

- entrypoint import, Hydra config, script helper 검증은 관련 root `tests/unit`
  케이스를 함께 갱신한다.
