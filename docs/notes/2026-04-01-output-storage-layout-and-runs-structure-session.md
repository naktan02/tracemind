# 2026-04-01 Output Storage Layout And Runs Structure Session

## User

이제 결과 나오는 위치 정리하고 싶은데 어떤건 data 폴더에 저장되고 어떤건 tmp에 저장되고 또 일부 main.log,run_federated_시뮬레이터 등 일부는 루트에 저장되는데 한곳으로 모는게 좋지 않을까?
data 폴더를 정립하면서 하는건 어때?

## Assistant

현재 활성 단계는 로컬 추론과 shared adapter FL 정리 단계라, 산출물 경계는 `shared/global artifact`, `local runtime state`, `ephemeral experiment output` 기준으로 나누는 게 맞다고 설명했다.

- `data/`는 재사용 가능한 입력/가공 산출물
- `state/`는 서버나 에이전트가 읽는 runtime/shared artifact
- `tmp/`와 루트 로그는 실험 실행 결과나 실행 습관 때문에 생긴 파일

전부 `data/`로 몰기보다 저장 정책을 먼저 정립하는 편이 맞다고 정리했다.

## User

evaluations/ 여기에는 뭐가 저장되는건데? 경계가 애매하지 않나?

## Assistant

`evaluations/` 아래 실제 파일과 설정을 확인한 뒤, 현재는 아래가 섞여 있다고 설명했다.

- 순수 평가 결과
- 시각화와 sweep grid 같은 분석 보조 산출물
- 전략 후보 index 같은 탐색용 산출물

그래서 `evaluations/`는 이름이 애매하고, 장기적으로는 `runs/` 아래 실행별 산출물로 보내는 편이 더 자연스럽다고 설명했다.

## User

스크립트 폴더는 이후에 리펙토링 진행할거고 산출물? 구조 정하는건데 이 구조엣서 이벨류에이션은 애매하지 않나?

```text
data/
  raw/
  processed/
  evaluations/
state/
  main_server/
    prototype_packs/
    shared_adapter_states/
    model_manifests/
  agents/
    <agent_id>/
      prototype_packs/
      shared_adapter_updates/
runs/
  federated_simulation/
    <run_id>/
      main_server/
  prototype_strategy/
    <run_id>/
      logs/
hf_cache/
```

## Assistant

`evaluations/`는 이름이 애매하니 top-level 개념으로 두지 않는 편이 낫다고 설명했다.

- `data/`는 재사용 가능한 입력과 산출물
- `state/`는 runtime 저장소
- `runs/`는 실행별 결과, 평가 리포트, 진단, 시각화, 로그

즉 평가 결과도 `runs/` 아래 실행 산출물로 보는 편이 더 덜 헷갈린다고 정리했다.

## User

이 부분은 뭐야?

```text
state/
  main_server/
    prototype_packs/
    shared_adapter_states/
    model_manifests/
  agents/
    <agent_id>/
      prototype_packs/
      shared_adapter_updates/
```

## Assistant

이 부분은 데이터 산출물 폴더가 아니라, 서버와 에이전트가 실제 실행 중에 읽고 갱신하는 file-based runtime store라고 설명했다.

- `main_server/...`는 중앙 서버 배포 원본과 shared adapter 상태
- `agent/...`는 로컬 캐시와 로컬 학습 update payload
- `model_manifests`는 값 자체보다 배포 메타데이터에 가깝다고 설명했다

## User

그럼 저건 각 에이전트나 서버의 실행 결과물이면 해당 폴더 내에 있는게 낫지 않아?

## Assistant

그 방향이 더 자연스럽다고 설명했다.

- `main_server/state/...`
- `agent/state/...`

처럼 각 컴포넌트 폴더 안에 두는 편이 소유권과 runtime 경계에 맞는다고 정리했다.

## User

main_server/
  state/
    prototype_packs/
    shared_adapter_states/
    model_manifests/

agent/
  state/
    prototype_packs/
    shared_adapter_updates/

이건 실제로 실행할 때 사용하는값이 들어가는건가?

## Assistant

큰 틀에서는 맞다고 답했다.

- `shared_adapter_states`, `prototype_packs`, `shared_adapter_updates`는 실행이 직접 읽거나 생성하는 값
- `model_manifests`는 그 값을 연결하는 배포 메타데이터

## User

그럼 저건 실제 값이라고 하고 밑에 느낌으로 가는건가?
runs에는 실험 결과들이 곗속 쌓이겠네federated_시뮬레이션이나 쓰레시 홀드등에도 process엣서 하는것처럼 해당 폴더 밑에 날짜로 구분하도록 하자 이 구조가 제일 좋을까?

```text
data/
  raw/
  processed/
    labeled_query_sets/
    splits/
    prototype_build_states/
    prototype_packs/
    classifier_heads/

runs/
  federated_simulation/
    <run_id>/
      main_server/
      agents/
      logs/
  prototype_pack_eval/
    <run_id>/
      report.json
  prototype_strategy/
    <run_id>/
      summary.json
      validation/
      projections/
      candidates/
      logs/
  prototype_threshold_sweep/
    <run_id>/
      summary.json
```

## Assistant

이 방향이 지금까지 나온 안들 중 가장 낫다고 설명했다.

- `data/processed`는 재사용 가능한 산출물
- `main_server/state`, `agent/state`는 실제 runtime 저장소
- `runs/<job>/<YYYY-MM-DD>/<run_id>`는 실행별 산출물

또 `run_manifest.json`을 두는 것도 장기적으로 유용하다고 덧붙였다.

## User

주의할점이 뭐라는거야? 총 정리해서 폴더 보여줘
그리고 main,agent state에서 프로토타입팩은 한쪽만 있어도 되는거 아냐? 어차치 계속 주고 받지 않나?

## Assistant

주의할 점을 아래처럼 정리했다.

1. `data`와 `runs`를 섞지 말 것
2. `state`는 각 컴포넌트 안에 둘 것
3. `runs`는 날짜 + run_id로 쌓을 것
4. 각 run은 나중에 설정 추적이 가능해야 할 것

그리고 `prototype_packs`는 서버/에이전트 양쪽에 모두 두는 것이 현재 구조상 맞다고 설명했다.

- 서버는 배포 원본
- 에이전트는 로컬 runtime cache

에이전트가 추론 시 서버에 매번 의존하지 않는 로컬 개인화 구조와 맞기 때문이라고 설명했다.

## User

이렇게 해서 진행하자

## Assistant

실제 변경 범위는 아래처럼 최소화하는 방향으로 정리했다.

- YAML 기본 경로 수정
- 날짜/run_id 경로 조립 함수 추가
- 관련 실험 스크립트 4개 이상에서 새 규칙 사용
- README와 ignore 정책 수정

## User

진행하는데 yaml 위치만 바꾸고 조금만 수정하면 끝 아냐?

## Assistant

`prototype_strategy`, `prototype_threshold_sweep` 쪽은 YAML 수정 비중이 크지만, 날짜/run_id 경로를 만들려면 작은 코드 수정이 함께 필요하다고 설명했다.

`run_federated_simulation`, `evaluate_prototype_pack`, `train_softmax_classifier`는 코드가 직접 출력 구조를 잡고 있어서 YAML만으로는 끝나지 않는다고 정리했다.

## User

그렇게해서 진행하자

## Assistant

실제 수정은 아래 방향으로 진행했다.

- `scripts/run_artifacts.py` 추가
  - `runs/<job>/<YYYY-MM-DD>/<run_id>` 경로 생성 유틸 추가
- `prototype_strategy`, `prototype_threshold_sweep`
  - 날짜/run_id 구조로 출력 경로 정규화
  - `logs/` 디렉터리 생성
- `evaluate_prototype_pack`
  - `runs/prototype_pack_eval/<date>/<run_id>/reports/` 구조로 변경
- `train_softmax_classifier`
  - 평가 리포트는 `runs/train_classifier/<date>/<classifier_version>/reports/`
  - 모델 아티팩트는 계속 `data/processed/classifier_heads/`
- `run_federated_simulation`
  - `runs/federated_simulation/<date>/<run_id>/...` 구조로 변경
- YAML 기본 경로 변경
  - `prototype_strategy`
  - `prototype_threshold_sweep`
  - `prototype_pack_eval`
  - `train_softmax_classifier`
  - `federated_run_preset`
- `.gitignore`
  - `main_server/state/` 경로로 수정
  - `runs/` 추가
- `data/README.md`, `scripts/README.md`
  - 새 저장 정책과 출력 구조 반영

## Assistant

검증으로 아래 테스트를 실행했고 통과했다.

```bash
uv run pytest \
  tests/unit/test_prototype_strategy_experiment.py \
  tests/unit/test_scripts_hydra_configs.py \
  tests/unit/test_run_federated_simulation.py
```

결과:

- `17 passed in 7.43s`

## Assistant

빈 디렉터리만 먼저 만들어두고, 기존 산출물은 사용자가 수동으로 옮기기로 정리했다.

생성한 디렉터리:

- `runs/federated_simulation/`
- `runs/prototype_pack_eval/`
- `runs/prototype_strategy/`
- `runs/prototype_threshold_sweep/`
- `runs/train_classifier/`
- `main_server/state/shared_adapter_states/versions/`
- `main_server/state/model_manifests/versions/`
- `agent/state/prototype_packs/versions/`
- `agent/state/shared_adapter_updates/`

수동 이동 기준:

- `data/processed/evaluations/prototype_strategy_experiments/<run_id>` -> `runs/prototype_strategy/<YYYY-MM-DD>/<run_id>`
- `data/processed/evaluations/prototype_threshold_sweeps/<run_id>` -> `runs/prototype_threshold_sweep/<YYYY-MM-DD>/<run_id>`
- `data/processed/evaluations/classifier_heads/<classifier_version>` -> `runs/train_classifier/<YYYY-MM-DD>/<classifier_version>/reports/`
- `data/processed/evaluations/prototype_packs/<prototype_version>` -> `runs/prototype_pack_eval/<YYYY-MM-DD>/<prototype_version>/reports/`
- `tmp/federated_simulation*` -> `runs/federated_simulation/<YYYY-MM-DD>/<manual_run_id>/`
- 루트 로그 파일은 대응하는 run의 `logs/` 아래로 수동 이동

주의:

- `data/processed/classifier_heads/` 아래 모델 파일은 그대로 유지
- `main_server/state/`, `agent/state/`는 런타임 저장소이므로 과거 실험 결과를 옮기는 용도로 쓰지 않음
