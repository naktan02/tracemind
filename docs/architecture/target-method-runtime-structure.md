# Target Method Runtime Structure

이 문서는 TraceMind의 최종 method/runtime 구조를 설명한다. 현재 코드와 shared
contract가 실제 실행 source of truth이지만, 앞으로의 구조 판단과 리팩터링 계획은
이 문서를 우선한다. 기존 active MD가 이 문서와 충돌하면 그 문서는 현행 상태나
legacy migration 기록으로 보고, 새 설계 판단에는 이 문서를 기준으로 삼는다.

## 목표

새 FL SSL 방법, classifier 기반 update, prototype 기반 update, PEFT mechanism을
추가할 때 변경 위치가 예측 가능해야 한다.

핵심 검증 기준:

```text
새 방법 추가 -> methods/와 conf/만 바뀌고 scripts runner는 그대로 둔다.
```

`scripts`가 `classifier`, `prototype`, `lora`, `fedmatch_agreement` 같은 method 또는
trainable state identity를 직접 분기하면 Seam이 얕은 구조로 본다.

## 최종 용어

| 용어 | 의미 | 예 |
|---|---|---|
| PEFT adapter mechanism | encoder에 붙는 PEFT mechanism | `lora`, `dora` |
| trainable state / update family | client가 학습하고 server가 해석하는 공유 가능 상태 family | `linear_head`, `peft_text_classifier`, `prototype_pack` |
| scorer family | local inference/training에서 score를 만드는 방식 | `classifier_logits`, `prototype_similarity` |
| method descriptor | FedMatch/FedLGMatch/(FL)^2 같은 논문 method identity와 local/server policy 요구사항 | `fedmatch` |
| runtime capability | 여러 method가 공유할 수 있는 실행 능력 | `peer_context`, `server_step`, `artifact_ref_materializer`, `security_policy` |

`adapter_family_name`은 현재 v1 코드와 contract에 남은 이름이다. 최종 구조에서는
prototype까지 포함하기 어렵기 때문에 `update_family_name` 또는
`trainable_state_family_name`으로 교체한다.

`text`가 붙은 이름은 의도적으로 modality-specific이다. `peft_text_classifier`는
text encoder, tokenizer, text view 생성, PEFT adapter, linear head가 함께 움직이는
update family다. 반대로 `linear_head`, `prototype_pack`, `classifier_logits`,
`prototype_similarity`는 embedding/vector 위에서 동작하는 modality-neutral primitive나
family로 둔다. 이미지나 오디오를 추가할 때는 text-specific family를 억지로 확장하지
않고 `vision_peft_classifier`, `audio_peft_classifier`, 또는 여러 modality에서 실제로
같은 Interface가 검증된 뒤의 `peft_embedding_classifier` 같은 새 update family를
`methods/`와 `conf/strategy_axes/trainable_state/update_family/`에 추가한다.
`scripts`는 이 차이를 이름으로 분기하지 않고 resolved execution plan만 실행한다.

## 최종 구조

```text
shared/
  contracts/
    trainable_state/{linear_head,peft_text_classifier,prototype_pack}.py
    trainable_state/legacy_lora_classifier_v1.py
    fl_ssl/{execution_plan,update_payload,report_protocol}.py

methods/
  classification/linear_head/
  prototype/{building,scoring,evidence,training_inputs}/
  prototype/aggregation_projection.py
  adaptation/
    peft_adapters/{lora,dora}/
    peft_text_classifier/{training,update,aggregation,federated_ssl}/
    query_text_views/
    privacy_guards/
    local_objective_regularizers/
  ssl/algorithms/{fixmatch,flexmatch,freematch,adamatch,pseudolabel}/
  ssl/hooks/
  federated/{aggregation/fedavg,participation.py,shard_policy}/
  federated_ssl/
    {execution_plan,capability_axes,compatibility_validator}.py
    manual_baseline/descriptor.py
    fedmatch/{descriptor,original_spec,local_objective,server_policy}.py
    fedlgmatch/
    fl2/

conf/
  entrypoints/
    {fl_ssl,central_ssl,prototype_analysis}/run.yaml
  strategy_axes/
    ssl/{consistency_method,pseudo_label_selection}/
    trainable_state/update_family/
    adaptation/{peft_adapter,transformer_backbone,initial_checkpoint}/
    prototype/build_strategy/
    fl/
      method_descriptor/
      local_update_profile/
      round_runtime_profile/
      runtime capability axes...

scripts/
  experiments/
    {fl_ssl,central_ssl,prototype_analysis}/{run,sweep,report}.py
  runtime_adapters/{federated_agent,federated_server,artifact_io,report_io}/
```

## 구조 규칙

아래 규칙은 compact tree보다 우선한다. 위 구조도는 최종 소유 경계와 이름을 보여주는
지도이며, 모든 파일을 한 번에 만들라는 구현 목록이 아니다.

1. `linear_head`는 작은 classification primitive다.
   `peft_text_classifier`와 같은 레벨의 큰 adaptation family로 키우지 않는다.
   PEFT text classifier 안에는 classifier head가 포함될 수 있지만, 그 family는
   text encoder, PEFT adapter, classifier head를 함께 학습하는 더 큰 Module이다.

2. `prototype_pack`은 LoRA 같은 adapter mechanism이 아니다.
   하지만 FL 관점에서는 client가 만들고 server가 해석하는 update family가 될 수
   있다. 따라서 prototype은 `adapter_family`가 아니라 `update_family` 또는
   `trainable_state` 축으로 다룬다.

3. `fedmatch_agreement` 같은 method-local objective는 generic config axis leaf로
   노출하지 않는다. `conf`에서는 `method_descriptor=fedmatch`만 고르고,
   `methods/federated_ssl/fedmatch/`가 agreement vote, sigma/psi, helper 요구사항을
   소유한다.

4. `fixed_probe_output_knn`, `server_step_policy`, `security_policy`처럼 여러 method가
   공유할 수 있는 mechanism은 runtime capability axis에 둘 수 있다. 반대로
   `num_helpers=2`, `h_interval`, `sigma/psi` partition 의미는 FedMatch method package가
   소유한다.

5. `scripts`의 Interface는 config load, execution plan resolve, runtime context 구성,
   core 호출, output write로 제한한다.
   `conf/strategy_axes/trainable_state/update_family/*`는 update family 이름과
   methods-owned initial state builder, validation evaluator, final projection builder
   path를 제공할 수 있다.
   `conf/strategy_axes/fl/server_step_policy/*`처럼 runtime capability leaf도
   필요하면 runtime adapter executor path를 선언할 수 있다.
   scripts는 이 callable을 import/execute하는 generic adapter만 소유하고,
   `linear_head`, `peft_text_classifier`, `diagonal_scale`, `prototype_pack` 같은
   family별 초기화/평가 분기나 `supervised_seed_step`의 PEFT 세부 구현을 직접
   소유하지 않는다.

```python
cfg = load_config()
plan = resolve_federated_ssl_execution_plan(cfg)
context = build_run_context(cfg)
result = execute_plan(plan, context)
write_outputs(result, context)
```

이 흐름에 method/update-family 분기가 들어가면 해당 분기는 `methods`나 runtime
Adapter 뒤로 옮긴다.

## 해석 Guard

- compact tree에 없는 helper 파일을 만들 수 있다. 단, 새 파일은 가장 가까운 owner
  Module에 들어가야 하고, method identity가 scripts, agent, main_server로 새면 안 된다.
- `methods/classification/linear_head`는 classifier 학습 runtime 전체가 아니라
  linear head의 scoring/bootstrap/projection primitive다.
- `methods/adaptation/peft_text_classifier`는 text encoder, PEFT mechanism, linear head를
  함께 학습하는 update family다. 여기서 LoRA는 mechanism이고 family 이름이 아니다.
- `methods/prototype/*`는 prototype pack 생성, scoring, evidence, training input,
  projection을 소유한다. prototype은 adapter mechanism이 아니라 update family 또는
  scorer family로 해석한다.
- `conf/strategy_axes/fl/method_descriptor/<method>.yaml`은 method identity 선택
  표면이다. method-local objective 이름을 별도 generic leaf로 다시 노출하지 않는다.
- `fedmatch_agreement`, `sigma/psi`, FedMatch helper 기본값은
  `methods/federated_ssl/fedmatch/`가 소유한다. 여러 method가 공유할 수 있는
  mechanism만 runtime capability axis로 승격한다.
- `scripts`는 새 method나 update family를 알기 위해 수정하지 않는다. runner 수정이
  필요하면 먼저 execution plan Interface가 충분히 깊은지 점검한다.
- 현행 문서나 코드의 `lora_classifier`, `adapter_family_name`은 compatibility 이름일 수
  있다. 새 구조 설계에서는 `peft_text_classifier`, `update_family_name`,
  `trainable_state` 용어로 해석한다.

## Legacy 격리

- `lora_classifier`는 v1 shared contract와 old artifact/import compatibility 이름이다.
  새 canonical update family 이름은 `peft_text_classifier`다.
- `diagonal_scale`은 현행 no-config runtime fallback과 테스트 fixture로 남아 있다.
  제거하려면 먼저 fallback을 명시 profile 기반으로 바꾸고 테스트용 fake family를
  분리한다.
- `round_runtime.adapter_family_name`은 현재 실행 field다. 최종 migration에서는
  `round_runtime.update_family_name` 또는 `trainable_state_family_name`으로 바꾼다.
- legacy field와 shim은 제거 조건이 있는 compatibility layer로만 유지한다.

## Migration Plan

### 0단계: target 문서 고정

- 이 문서를 active 구조 판단 기준으로 등록한다.
- 기존 `text-classifier-adaptation-refactor-plan.md`와 `strategy_surface_map.md`는
  현행 상태와 migration history로 읽고, 최종 구조 판단은 이 문서를 따른다.

### 1단계: execution plan Interface 깊게 만들기

- `methods/federated_ssl/execution_plan.py`가 method descriptor, update family,
  runtime capability 요구사항을 typed plan으로 resolve한다.
- Compatibility validator는 scripts 실행 중간이 아니라 bootstrap 전에 실패하게 한다.
- 테스트는 `method_descriptor=fedmatch`가 `fedmatch_agreement`를 내부 요구사항으로
  파생하지만 generic `local_ssl_policy` override로 노출하지 않는 것을 검증한다.

### 2단계: config vocabulary 전환

- `conf/strategy_axes/trainable_state/update_family/`를 추가한다.
- 기존 `round_runtime.adapter_family_name`은 compatibility alias로 읽되, 새 config와
  report에는 `update_family_name`을 병행 기록한다.
- `strategy_axes/fl/local_ssl_policy/fedmatch_agreement.yaml` 같은 method-local leaf는
  method descriptor 내부 요구사항으로 이동한다.

### 3단계: methods 구조 정리

- `methods/adaptation/classification/feature_head`의 최종 위치를
  `methods/classification/linear_head`로 정한다.
- `methods/adaptation/text_classifier/peft_encoder`의 canonical 이름을
  `methods/adaptation/peft_text_classifier`로 전환한다.
- `methods/adaptation/query_classifier_adaptation`은 `query_text_views`로 바꿔
  input/view glue 역할을 이름에 드러낸다.
- `lora_classifier` shim은 v1 compatibility package로 축소한다.

### 4단계: scripts 분기 제거

- FL/central runner가 update family나 method 이름으로 분기하는 지점을 제거한다.
- scripts runtime adapter는 artifact load, request mapping, repository wiring만 맡긴다.
- 새 classifier/prototype method를 추가해도 `scripts/experiments/fl_ssl/run.py`가
  바뀌지 않는 architecture test를 둔다.

### 5단계: shared contract v2

- `shared/contracts/trainable_state/*`에 canonical payload family를 둔다.
- v1 `lora_classifier` contract는 `legacy_lora_classifier_v1`로 격리하고 제거 조건을
  `legacy_contract_ledger`에 연결한다.
- report protocol은 current legacy name과 target canonical name을 migration window
  동안 함께 기록한다.

### 6단계: compatibility 제거

- old run/report/dashboard v2 single-write가 안정되면 legacy config alias,
  direct import shim, old contract field를 순서대로 제거한다.
- 제거는 artifact compatibility audit와 architecture guard가 통과한 뒤 진행한다.

## Definition Of Done

prototype 기반 FL SSL method를 추가할 때 변경 위치가 아래로 닫히면 성공이다.

```text
methods/prototype/*
methods/federated_ssl/<method>/*
conf/strategy_axes/fl/method_descriptor/<method>.yaml
conf/strategy_axes/fl/local_update_profile/prototype_*.yaml
conf/strategy_axes/trainable_state/update_family/prototype_pack.yaml
tests/*
```

그리고 아래 파일은 바뀌지 않아야 한다.

```text
scripts/experiments/fl_ssl/run.py
```

이 조건이 깨지면 Module의 Interface가 충분히 깊지 않고, method/update-family Seam이
runtime orchestration 쪽으로 새고 있는 것으로 본다.
