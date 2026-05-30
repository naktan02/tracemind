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
| trainable state / update family | client가 학습하고 server가 해석하는 공유 가능 상태 family | `linear_head`, `peft_text_encoder`, `prototype_pack` |
| scorer family | local inference/training에서 score를 만드는 방식 | `classifier_logits`, `prototype_similarity` |
| method descriptor | FedMatch/FedLGMatch/(FL)^2 같은 논문 method identity와 local/server policy 요구사항 | `fedmatch` |
| runtime capability | 여러 method가 공유할 수 있는 실행 능력 | `peer_context`, `server_step`, `artifact_ref_materializer`, `security_policy` |

`adapter_family_name`은 제거된 v1 입력 이름이다. 새 실행 config, runtime model,
report/result reader는 `payload_adapter_kind`와 `update_family_name`만 받는다.
prototype까지 포함하는 실행 축은 `update_family_name` 또는
`trainable_state_family_name`으로 표현한다.

`text`가 붙은 이름은 의도적으로 modality-specific이다. `peft_text_encoder`는
text encoder, tokenizer, text view 생성, PEFT adapter, head/scorer가 함께 움직이는
update family다. 현재 기본 head primitive는 `linear_head`지만 runner와 report는
그 사실로 분기하지 않는다. 반대로 `linear_head`, `prototype_pack`,
`classifier_logits`, `prototype_similarity`는 embedding/vector 위에서 동작하는
modality-neutral primitive나 family로 둔다. 이미지나 오디오를 추가할 때는 text-specific family를 억지로 확장하지
않고 `vision_peft_classifier`, `audio_peft_classifier`, 또는 여러 modality에서 실제로
같은 Interface가 검증된 뒤의 `peft_embedding_classifier` 같은 새 update family를
`methods/`와 `conf/strategy_axes/model_architecture/update_family/`에 추가한다.
`scripts`는 이 차이를 이름으로 분기하지 않고 resolved execution plan만 실행한다.

## 최종 구조

```text
shared/
  contracts/
    trainable_state/{linear_head,peft_text_encoder,prototype_pack}.py
    fl_ssl/{execution_plan,update_payload,report_protocol}.py

methods/
  classification/linear_head/
  prototype/{building,scoring,thresholding,evidence,training_inputs}/
  prototype/aggregation_projection.py
  adaptation/
    peft_adapters/{lora,dora}/
    peft_text_encoder/{training,update,aggregation,federated_ssl}/
    query_text_views/
    privacy_guards/
    local_objective_regularizers/
  ssl/algorithms/{fixmatch,flexmatch,freematch,adamatch,pseudolabel}/
  ssl/hooks/
  federated/{aggregation/fedavg,participation.py,shard_policy}/
  federated_ssl/
    {execution_plan,capability_axes,compatibility_validator}.py
    manual_baseline/descriptor.py
    fedmatch/{descriptor,method_surface,original_spec,local_objective}.py
    fedlgmatch/
    fl2/

conf/
  entrypoints/
    {fl_ssl,central_ssl,prototype_analysis}/run.yaml
  strategy_axes/
    ssl_objective/{consistency_method,local_update_profile}/
    model_architecture/{backbone,trainable_surface,peft,update_family,initial_checkpoint}/
    prototype/build_strategy/
    fssl_method/
    fl_topology/
      {shard_policy,labeled_exposure,client_participation,materialized_split}/
      {server_step,server_update,peer_context,update_partition,aggregation_weight}/
  run_controls/
    fl_ssl/{budget,safety_and_sweeps}/
  execution_context/
    {dataset_asset,query_data_source,embedding_adapter,runtime_env,security_policy}/

scripts/
  experiments/
    {fl_ssl,central_ssl,prototype_analysis}/{run,sweep,report}.py
  runtime_adapters/{federated_agent,federated_server,artifact_io,report_io}/
```

## 구조 규칙

아래 규칙은 compact tree보다 우선한다. 위 구조도는 최종 소유 경계와 이름을 보여주는
지도이며, 모든 파일을 한 번에 만들라는 구현 목록이 아니다.

1. `linear_head`는 작은 classification primitive다.
   `peft_text_encoder`와 같은 레벨의 큰 adaptation family로 키우지 않는다.
   PEFT text encoder 안에는 현재 linear head가 포함될 수 있지만, 그 family는
   text encoder, PEFT adapter, head/scorer를 함께 학습하는 더 큰 Module이다.
   scripts와 report는 head 구현이 linear인지 prototype인지 직접 알지 않는다.

2. `prototype_pack`은 LoRA 같은 adapter mechanism이 아니다.
   하지만 FL 관점에서는 client가 만들고 server가 해석하는 update family가 될 수
   있다. 따라서 prototype은 `adapter_family`가 아니라 `update_family` 또는
   `trainable_state` 축으로 다룬다.

3. `fedmatch_agreement` 같은 method-local objective는 generic config axis leaf로
   노출하지 않는다. `conf`에서는 `fssl_method=fedmatch`로 method identity를 고르고,
   `ssl_method.scenario=labels-at-client` 또는 `labels-at-server`로 label 위치를
   고른다. `methods/federated_ssl/fedmatch/`가 agreement vote, sigma/psi, helper
   요구사항과 scenario별 policy default를 소유한다.

4. `fixed_probe_output_knn`, `server_step_policy`, `security_policy`처럼 여러 method가
   공유할 수 있는 mechanism은 runtime capability axis에 둘 수 있다. 반대로
   `num_helpers=2`, `h_interval`, `sigma/psi` partition 의미는 FedMatch method package가
   소유한다.

5. `scripts`의 Interface는 config load, execution plan resolve, runtime context 구성,
   core 호출, output write로 제한한다.
  `conf/strategy_axes/model_architecture/update_family/*`는 update family 이름과
  runtime payload key, methods-owned initial state builder, validation evaluator,
  final projection builder, transient resource cleaner path를 제공할 수 있다.
  runtime payload 일부를 TrainingTask objective extra로 전달해야 하는 family는
  objective payload scope와 제외 key를 같은 update-family leaf에 선언한다.
  `conf/strategy_axes/fl_topology/server_step/*`처럼 runtime capability leaf도
  필요하면 runtime adapter executor path를 선언할 수 있다.
   scripts는 이 callable을 import/execute하는 generic adapter만 소유하고,
   `linear_head`, `peft_text_encoder`, `prototype_pack` 같은
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

6. 중앙 SSL의 teacher는 독립 실행 stage나 별도 strategy axis가 아니다.
   방법론 또는 내부 workflow/helper가 teacher 사용 여부를 내부에서 결정한다.
   checkpoint artifact에서 pseudo-label input을 materialize하는 bootstrap helper는
   public experiment entrypoint가 아니라 내부 compatibility workflow로만 남기고,
   fixed embedding classifier는 그 helper의 내부 artifact kind로 manifest에만 남긴다.
   prototype, PEFT checkpoint, EMA teacher, external pseudo-label source는 별도
   user-facing teacher selector가 아니라 새 method/recipe 또는 method descriptor
   내부 요구사항으로 추가한다. scripts는 teacher 구현 family를 새 axis로 노출하지
   않고, artifact materialization, Hydra cfg 해석, output write 같은 실행 glue만
   맡는다.

## 해석 Guard

- compact tree에 없는 helper 파일을 만들 수 있다. 단, 새 파일은 가장 가까운 owner
  Module에 들어가야 하고, method identity가 scripts, agent, main_server로 새면 안 된다.
- `methods/classification/linear_head`는 classifier 학습 runtime 전체가 아니라
  linear head의 scoring/bootstrap/projection primitive다.
- `methods/adaptation/peft_text_encoder`는 text encoder, PEFT mechanism, head/scorer를
  함께 학습하는 update family다. 여기서 LoRA는 mechanism이고 family 이름이 아니다.
- `methods/prototype/*`는 prototype pack 생성, scoring, thresholding, evidence,
  training input, projection을 소유한다. prototype은 adapter mechanism이 아니라
  update family 또는 scorer family로 해석한다.
- `conf/strategy_axes/model_architecture/update_family/*`는 update family가 필요로 하는
  runtime payload key와 runtime adapter callable을 선언한다. local objective executor,
  initial state builder, validation evaluator, final projection builder, transient
  resource cleaner는 scripts가 family 구현을 직접 import하지 않기 위한 설정 표면이다.
  migration window 동안 v1 shared payload가 필요한 family도 새 config leaf에서는
  `payload_adapter_kind`만 선언한다. `adapter_family_name` fallback은 report/result
  reader에서도 제거된 상태를 유지한다.
- `conf/strategy_axes/fssl_method/<method>.yaml`은 method identity 선택
  표면이다. method-local objective 이름을 별도 generic leaf로 다시 노출하지 않는다.
- `fedmatch_agreement`, `sigma/psi`, FedMatch helper 기본값은
  `methods/federated_ssl/fedmatch/`가 소유한다. 여러 method가 공유할 수 있는
  mechanism만 runtime capability axis로 승격한다.
- `scripts`는 새 method나 update family를 알기 위해 수정하지 않는다. runner 수정이
  필요하면 먼저 execution plan Interface가 충분히 깊은지 점검한다.
- `fixed_classifier_seed`는 active 실행 축이 아니다. classifier/prototype/PEFT 같은
  teacher source는 method/recipe 내부 요구사항과 checkpoint/artifact reference로
  표현하고, 별도 entrypoint/config stage나 user-facing teacher selector로 되살리지
  않는다.
- 현행 문서나 old artifact의 `lora_classifier`, `adapter_family_name`은 historical
  이름이다. 새 구조 설계, 새 실행 config, report/result reader에서는
  `peft_text_encoder`, `payload_adapter_kind`, `update_family_name`,
  `trainable_state` 용어로 해석한다.

## Legacy 격리

- `lora_classifier`는 제거된 v1 이름이다. shared parser/factory와 report/result
  reader fallback은 제거됐고, 새 canonical update family 이름은
  `peft_text_encoder`, payload kind는 `peft_classifier`다.
- `diagonal_scale`은 제거된 v1 payload family 이름이다. methods-level 구현 core,
  shared contract parser/factory, 전용 unit test, runtime fallback 기본값,
  privacy guard 실행 등록은 제거됐다. `diagonal_scale`는 target update-family 축이
  아니므로 `conf/strategy_axes/model_architecture/update_family/diagonal_scale.yaml`,
  `methods/adaptation/diagonal_scale/**`,
  `shared/src/contracts/adapter_contract_families/diagonal_scale.py`는 제거된 상태를
  유지한다.
- `round_runtime.adapter_family_name`은 제거된 v1 입력 이름이다. 새 config leaf,
  runtime model, report/result reader는 이 값을 생산하거나 받지 않는다. 새 실행
  표면은 `round_runtime.update_family_name`과
  `round_runtime.payload_adapter_kind`를 기준으로 한다.
- legacy field는 제거 조건이 있는 compatibility layer로만 유지한다. 삭제된
  methods-level direct import shim은 다시 만들지 않는다.

## Migration Plan

### 0단계: target 문서 고정

- 이 문서를 active 구조 판단 기준으로 등록한다.
- 기존 text-classifier migration plan archive와 `strategy_surface_map.md`는 현행
  상태와 migration history로 읽고, 최종 구조 판단은 이 문서를 따른다.

### 1단계: execution plan Interface 깊게 만들기

- `methods/federated_ssl/execution_plan.py`가 method descriptor, update family,
  runtime capability 요구사항을 typed plan으로 resolve한다.
- Compatibility validator는 scripts 실행 중간이 아니라 bootstrap 전에 실패하게 한다.
- 테스트는 FedMatch descriptor와 scenario default가 `fedmatch_agreement`를 내부
  요구사항으로 파생하지만 generic `local_ssl_policy` override로 노출하지 않는 것을
  검증한다.

### 2단계: config vocabulary 전환

- `conf/strategy_axes/model_architecture/update_family/`를 추가한다.
- 기존 `round_runtime.adapter_family_name` 입력 정규화는 제거됐다. 새 config,
  report producer, report/result reader는 `update_family_name`과
  `payload_adapter_kind`만 생산하고 읽는다.
- `fedmatch_agreement` 같은 method-local objective는 generic
  `strategy_axes/ssl_objective/local_ssl_policy/*.yaml` leaf가 아니라 method descriptor 내부
  요구사항에서 파생한다.

### 3단계: methods 구조 정리

- `methods/classification/linear_head`가 현재 modality-independent linear classifier
  head primitive의 canonical 구현 위치다. 새 head/scorer 방식이 추가되어도
  scripts/report가 linear 이름으로 바뀌지 않아야 한다.
- `methods/adaptation/peft_text_encoder`가 PEFT text encoder update family의
  canonical 구현 위치다.
- `methods/adaptation/query_text_views`는 text query input/view glue 역할만
  소유한다.
- 삭제된 `methods/adaptation/lora_classifier` direct import path는 다시 만들지
  않는다. v1 `lora_classifier` 이름은 old artifact reader, report compatibility,
  materialization fallback처럼 실제 경계가 있는 표면에만 남긴다.

### 4단계: scripts 분기 제거

- FL/central runner가 update family나 method 이름으로 분기하는 지점을 제거한다.
- scripts runtime adapter는 artifact load, request mapping, repository wiring만 맡긴다.
- 새 classifier/prototype method를 추가해도 `scripts/experiments/fl_ssl/run.py`가
  바뀌지 않는 architecture test를 둔다.

### 5단계: shared contract v2

- `shared/contracts/trainable_state/*`에 canonical payload family를 둔다.
- v1 `lora_classifier` shared contract와 report/result reader fallback은 제거된
  상태를 유지한다. 과거 tensor checkpoint를 반드시 읽어야 하는 흐름은
  materialization old-reader가 자기 경계에서 legacy tensor key를 읽고 canonical PEFT
  representation으로 정규화한다.
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
conf/strategy_axes/fssl_method/<method>.yaml
conf/strategy_axes/ssl_objective/local_update_profile/prototype_*.yaml
conf/strategy_axes/model_architecture/update_family/prototype_pack.yaml  # runtime callable까지 구현될 때 추가
tests/*
```

그리고 아래 파일은 바뀌지 않아야 한다.

```text
scripts/experiments/fl_ssl/run.py
```

이 조건이 깨지면 Module의 Interface가 충분히 깊지 않고, method/update-family Seam이
runtime orchestration 쪽으로 새고 있는 것으로 본다.
