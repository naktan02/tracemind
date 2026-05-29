# Strategy Surface Refactor Plan

이 문서는 중앙 SSL과 FL SSL 실행 표면에서 method 의미, scaffold 의미, 실험 조건
의미가 섞여 나온 축을 정리하는 active 계획이다. 과거 migration 기록이 아니라 현재
리팩터링 우선순위와 완료 기준만 유지한다.

최종 vocabulary는
`docs/architecture/target-method-runtime-structure.md`,
현재 축 지도는 `docs/strategy_surface_map.md`를 우선한다. 이 문서는 그 둘 사이에서
"어떤 축을 public surface에 남기고 무엇을 method 내부로 내릴지"를 작업 계획으로
구체화한다.

## 목표

1. 방법론을 고르면 method-owned 의미가 함께 닫히게 만든다.
2. scaffold 축과 실험 조건 축은 method 의미와 분리해 유지한다.
3. compatibility workflow나 bootstrap 절차를 strategy axis처럼 노출하지 않는다.
4. manual baseline에서만 lower-axis 조합을 직접 고를 수 있게 한다.

## 분류 기준

- Scaffold axis:
  학습 표면, backbone, adapter mechanism, update family처럼 바뀌면 실제
  trainable state, artifact kind, payload 의미가 바뀌는 축이다.
  예: `backbone`, `peft`, `trainable_surface`, `update_family`,
  `initial_checkpoint`
- Experiment condition axis:
  데이터 노출 조건, split, client 참여 규칙처럼 방법론 바깥의 비교 조건이다.
  예: `shard_policy`, `materialized_split`, `labeled_exposure`,
  `supervision_regime`, `client_participation`
- Method-owned axis:
  방법론을 고르면 같이 따라와야 하는 local objective, teacher usage, helper
  usage, server/round policy 의미다.
  예: `local_ssl_policy`, `server_update`, `peer_context`, `server_step`,
  `update_partition`, `aggregation_weight`
- Workflow / compatibility surface:
  방법론이 아니라 artifact materialization, bootstrap, replay 같은 절차다.
  예: `input_mode`, `teacher_bootstrap`, `teacher_*`, standalone
  pseudo-label materialization policy

## 리팩터링 대상 축

| 번호 | 현재 축/표면 | 현재 위치 | 문제 | 목표 상태 | 우선순위 |
|---|---|---|---|---|---|
| 1 | `input_mode` | `conf/strategy_axes/ssl_objective/input_mode/*` | method가 아니라 runner/workflow를 고르게 한다 | public strategy axis에서 제거. method recipe 또는 internal workflow/helper로 강등 | 높음 |
| 2 | `trainable_surface` | `conf/strategy_axes/model_architecture/trainable_surface/*` | 현재 구현이 하나뿐이라 축이 premature다 | 당장은 유지 가능. 단 실제 surface가 2개 이상 열릴 때까지 scaffold axis로만 취급 | 중간 |
| 3 | `pseudo_label_selection` | `conf/strategy_axes/ssl_objective/pseudo_label_selection/*` | 중앙 SSL method 축이 아니라 bootstrap/replay 내부 policy에 가깝다 | public method axis에서 제거. recipe 내부 기본값 또는 ablation metadata로 이동 | 높음 |
| 4 | `augmentation_source` | `conf/strategy_axes/ssl_objective/augmentation_source/*` | 일부 method에만 의미가 있을 수 있다 | 공통 input materialization surface면 유지, 특정 method 전용이면 method 내부로 이동 | 중간 |
| 5 | `teacher_*` bootstrap knobs | `teacher_bootstrap.yaml`, `scripts/support/query_ssl_peft/runners/teacher_source.py` | teacher 구현과 bootstrap IO가 user-facing 설정으로 과도하게 노출된다 | compatibility workflow 내부 값으로 강등. public central SSL surface에서 제거 | 높음 |
| 6 | `update_partition` | `conf/strategy_axes/fl_topology/update_partition/*` | method-owned 실행에서 method가 결정해야 할 partition semantics가 밖에 나온다 | manual baseline 전용 축으로 제한. method-owned에서는 descriptor/scenario가 파생 | 높음 |
| 7 | `aggregation_weight` | `conf/strategy_axes/fl_topology/aggregation_weight/*` | method-owned server policy의 일부가 public override로 남아 있다 | manual baseline 전용 축으로 제한. method-owned에서는 method server policy가 파생 | 높음 |
| 8 | `peer_context` | `conf/strategy_axes/fl_topology/peer_context/*` | helper context 사용 여부가 method recipe 일부인데 public surface에 남아 있다 | manual baseline 전용 축 또는 method-owned hidden capability | 높음 |
| 9 | `server_step` | `conf/strategy_axes/fl_topology/server_step/*` | labels-at-server / bootstrap semantics가 method scenario와 분리돼 있다 | manual baseline 전용 축 또는 method-owned scenario 파생값 | 높음 |
| 10 | `server_update` | `conf/strategy_axes/fl_topology/server_update/*` | method-owned server update semantics가 lower-axis로 노출돼 있다 | manual baseline 전용 축. method-owned는 descriptor default/variant가 소유 | 높음 |
| 11 | `local_update_profile` | `conf/strategy_axes/ssl_objective/local_update_profile/*` | method-owned에서 지원 조합이 사실상 고정인데 독립 축처럼 보인다 | manual baseline은 유지, method-owned는 method variant/scaffold compatibility로 흡수 | 높음 |
| 12 | `local_ssl_policy` | `conf/strategy_axes/ssl_objective/local_ssl_policy/*` | method-owned에서 실제로는 descriptor가 덮거나 제한하는 가짜 선택지가 된다 | manual baseline 전용 축. method-owned에서는 public surface에서 제거 | 높음 |
| 13 | `consistency_method` | `conf/strategy_axes/ssl_objective/consistency_method/*` | 중앙 SSL에선 핵심 method 축이지만 FSSL method-owned에선 중복될 수 있다 | central SSL과 manual baseline에서는 유지. FSSL method-owned에서는 hidden 또는 ignored | 중간 |

## 축별 목표 구조

### 중앙 SSL

사용자가 고를 수 있는 public surface는 아래 수준으로 줄인다.

- `backbone`
- `peft` mechanism
- `trainable_surface`
- `initial_checkpoint`
- central SSL method 또는 method recipe
- dataset/view/budget

중앙 SSL에서 내려야 하는 값:

- `input_mode`
- `teacher_bootstrap` 자체를 strategy axis로 쓰는 방식
- `teacher_*` runtime/training knobs
- standalone `pseudo_label_selection`

정리 방향:

1. `consistency`, `pseudo_label_replay`, `teacher_bootstrap`은 같은 레벨의 전략 축이 아니라
   method recipe 또는 workflow helper로 해석한다.
2. teacher는 독립 selector가 아니라 method recipe 내부 artifact source로 해석한다.
3. bootstrap이 필요하면 별도 workflow entrypoint 또는 이름 있는 method recipe로 승격한다.

### FL SSL manual baseline

manual baseline에서는 lower-axis 조합을 계속 허용한다.
남길 수 있는 값은 `consistency_method`, `local_ssl_policy`,
`local_update_profile`, `server_update`, `aggregation_weight`,
`update_partition`, `peer_context`, `server_step`이다. 단, 이 값들은 method 비교가
아니라 baseline/ablation 조합으로만 해석한다.

### FL SSL method-owned

method-owned에서는 사용자가 `fssl_method`, scenario/parameter override, split/exposure/
supervision/client participation, scaffold axis만 고르게 만든다.
`local_ssl_policy`, `local_update_profile`, `server_update`, `server_step`,
`peer_context`, `update_partition`, `aggregation_weight`, 경우에 따라
`consistency_method`는 숨기고 descriptor, method config surface, scenario preset이
파생한다.

## method variant 규칙

아래 중 하나라도 바뀌면 lower-axis override보다 새 method/variant 이름으로
분리하는 쪽을 기본으로 한다.

- pseudo-label 생성 규칙의 핵심 의미
- helper agreement 사용 여부
- supervised/unsupervised loss routing
- partition 해석
- server-client coupling semantics

예: `fedmatch`, `fedmatch_fixmatch_local`, `fedmatch_no_helper`,
`fedmatch_labels_at_server`

반대로 같은 method 안 override로 남길 수 있는 값은 helper 수, refresh interval,
threshold 수치, budget policy, 논문 원본 파라미터 사용 여부다.

## 단계별 계획

### Phase 0. 문서와 기준 고정

1. active docs에서 네 분류 기준을 명시한다.
2. central SSL, FSSL README 예시에서 method-owned 실행이 lower-axis를 같이 고르도록
   유도하는 문장을 제거할 준비를 한다.
3. architecture guard와 Hydra compose test에서 "method-owned hidden axis" 기준을
   검증할 위치를 정한다.

완료 기준:

- 현재 13개 축이 어떤 분류에 속하는지 active 문서에서 한 곳에 보인다.

### Phase 1. 중앙 SSL workflow surface 강등

1. `input_mode`를 strategy axis가 아니라 workflow/helper layer로 내린다.
2. `teacher_bootstrap`을 central SSL public surface에서 제거한다.
3. `teacher_*` 설정과 `pseudo_label_selection`을 bootstrap workflow 내부 설정으로
   이동한다.
4. central SSL public method surface를 다시 정의한다.

완료 기준:

- 중앙 SSL entrypoint 사용자가 teacher 구현, bootstrap split, bootstrap optimizer
  knob를 직접 고르지 않는다.

### Phase 2. FSSL method-owned lower-axis 숨김

1. `method_owned`일 때 `local_ssl_policy`, `local_update_profile`, `server_update`,
   `server_step`, `peer_context`, `update_partition`, `aggregation_weight`가 public
   surface에서 의미를 갖지 않게 한다.
2. descriptor/scenario/method config surface가 이 값을 파생한다.
3. manual baseline에서만 lower-axis 조합을 계속 허용한다.

완료 기준:

- `method_owned + fssl_method=fedmatch` 실행 예시에서 lower-axis override가 필요 없다.

### Phase 3. method variant surface 정리

1. FedMatch 계열에서 local objective나 helper semantics를 바꾸는 실험은
   `local_ssl_policy` override가 아니라 method variant 이름으로 승격한다.
2. method config surface는 parameter override와 scenario만 노출한다.
3. 보고서에는 manual baseline과 method-owned variant를 명확히 구분해 남긴다.

완료 기준:

- "FedMatch인데 local SSL policy만 바꿨다" 같은 애매한 실행 이름이 사라진다.

### Phase 4. Scaffold axis 재검토

1. `trainable_surface`가 실제로 두 개 이상 구현/검증되면 public scaffold axis로 유지한다.
2. `augmentation_source`는 공통 materialization surface인지, 특정 method 전용 입력인지
   다시 판정한다.
3. `consistency_method`는 central SSL과 manual baseline에는 유지하고,
   method-owned FSSL에서의 노출 여부를 정리한다.

완료 기준:

- scaffold axis는 실제로 독립 비교 가능한 값만 남는다.

## 검증 계획

코드 변경 단계에서 최소한 아래 검증이 함께 따라와야 한다.

1. Hydra compose test
   - central SSL public surface가 줄어든 뒤도 기본 compose가 유지되는지
   - method-owned FSSL에서 hidden axis가 노출되지 않거나 무시되는지
2. architecture guard
   - `scripts`가 method 이름이나 teacher 구현 이름으로 분기하지 않는지
   - method-owned path가 lower-axis public config에 다시 의존하지 않는지
3. report verification
   - manual baseline과 method-owned variant가 report에서 구분되는지
4. code-adjacent README sync
   - central SSL / FL SSL 실행 예시가 새 public surface와 맞는지

## 비목표

- 지금 문서만으로 최종 method taxonomy를 확정하지 않는다.
- 당장 `trainable_surface`를 제거한다고 결정하지 않는다.
- teacher compatibility code를 즉시 삭제하지 않는다.
- manual baseline 조합 능력 자체를 없애지 않는다.

## 현재 우선순위

1. 중앙 SSL의 `input_mode`, `teacher_*`, `pseudo_label_selection` 강등
2. FSSL `method_owned`에서 lower-axis hidden/derived화
3. FedMatch 계열 변형을 method variant 이름으로 다루는 규칙 정착
4. 마지막에 scaffold axis (`trainable_surface`, `augmentation_source`,
   `consistency_method`)의 잔여 공개 표면 재판정
