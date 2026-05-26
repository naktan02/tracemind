# Text Classifier Adaptation Refactor Plan

이 문서는 `methods/adaptation/lora_classifier/` 중심 구조를 장기적으로
교체 가능한 text classifier adaptation 구조로 바꾸기 위한 계획이다. 지금 단계에서는
코드를 이동하지 않고, 다음 리팩터링의 목표 구조와 검증 기준만 고정한다.

## 목적

현재 `lora_classifier` 이름은 task model, adaptation variant, PEFT mechanism,
FL aggregation, family projection, FL SSL method semantics를 한 폴더에 묶는다.
최종 목표는 새 method, 새 PEFT mechanism, 새 classifier variant를 추가할 때 수정
위치가 예측 가능하고, FedMatch 같은 method 의미가 runtime primitive나 modeling
파일에 강결합되지 않는 구조다.

## 핵심 결정

1. `classifier_head/`와 `lora/`를 같은 레벨의 sibling으로 계속 키우지 않는다.
   `classifier_head`는 task output/head 축이고, `lora`는 PEFT mechanism 축이다.
2. `classifier_head/`와 `peft_text_classifier/`도 sibling으로 키우지 않는다.
   classifier-head 자체는 modality-independent classification primitive이고,
   PEFT text encoder는 text-specific adaptation variant다.
3. modality-independent classifier-head 구현은 `methods/adaptation/classification/`
   아래로 두고, text encoder 관련 구현은 `methods/adaptation/text_classifier/`
   아래로 모은다.
4. LoRA, DoRA 같은 encoder adapter mechanism은
   `methods/adaptation/peft_adapters/` 아래에 둔다.
5. FedAvg algorithm은 `methods/federated/aggregation/fedavg/`가 소유하고,
   classification/text_classifier aggregation 계층은 FedAvg 입력/출력 projection만
   소유한다.
6. FedMatch method semantics는 `methods/federated_ssl/fedmatch/`에 남기고,
   adaptation 쪽 `federated_ssl/`은 method-neutral primitive만 소유한다.
7. shared contract 이름과 runtime report field는 마지막에 바꾼다.
   기존 `adapter_kind=lora_classifier`, `payload_format=lora_classifier_update`는
   compatibility가 끝날 때까지 유지한다.

## 목표 구조

```text
methods/adaptation/
  text_classifier/
    README.md

    common/
      labels.py
      logits.py
      metrics.py
      tensor_state.py
      runtime_config.py
    peft_encoder/
      README.md
      config.py
      modeling.py
      initial_state.py
      evaluation.py
      runtime_compatibility.py
      server_preflight.py
      training/
        batching.py
        delta_extraction.py
        loops.py
        optimizer_step.py
        pseudo_label_diagnostics.py
        query_ssl_local_training.py
        scalar_metrics.py
        step_budget.py
      update/
        delta_artifacts.py
        local_update.py
        materialization.py
        merged_tensor_artifact.py
        partitioned_delta.py
        partitioned_payload_builder.py
        partitioned_tensor_artifact.py
        payload_builder.py
        query_ssl_update.py
        simulation_inline_delta.py
      federated_ssl/
        README.md
        method_owned_training.py
        helper_provider.py
        peer_predictions.py
        supervised_seed_step.py
        partitioned/
          budget.py
          model_builder.py
          sparse_sync.py
          trainable_model.py
          training_loop.py
    aggregation/
      peft_encoder_fedavg_projection.py
      peft_encoder_partitioned_projection.py
      peft_encoder_state_projection.py

  classification/
    README.md
    feature_head/
      README.md
      bootstrap.py
      scoring.py
    aggregation/
      README.md
      feature_head_fedavg_projection.py
      feature_head_state_projection.py

  peft_adapters/
    README.md
    base.py
    registry.py
    lora/
      README.md
      builder.py
      target_modules.py
    dora/
      README.md
      builder.py
  diagonal_scale/
  privacy_guards/
  local_objective_regularizers/
  query_classifier_adaptation/
```

`peft_encoder/`는 순수 encoder만 뜻하지 않는다. 이 폴더는
“PEFT-adapted encoder + task classifier head” adaptation variant를 소유한다.
LoRA/DoRA 같은 순수 PEFT mechanism 구현은 `peft_adapters/`가 소유한다.

`classification/aggregation/*`와 `text_classifier/aggregation/*`는 family projection
계층이다. 여기서 weighted average, client weighting, global aggregation policy를 새로
구현하지 않는다. 그런 algorithm은 `methods/federated/aggregation/` 아래 generic
aggregation core가 소유한다.

FedMatch 쪽은 `methods/federated_ssl/fedmatch/`가 descriptor, original spec,
parameter routing, helper policy, local objective, server policy를 소유한다.
`objective/`나 `partition_plan.py` 같은 세부 분리는 `local_objective.py`가
과도하게 커지거나 FedLGMatch와 공유할 seam이 실제로 생길 때만 진행한다.

## 기존 파일 매핑

| 현재 위치 | 목표 위치 | 비고 |
|---|---|---|
| `methods/adaptation/lora_classifier/config.py` | `methods/adaptation/text_classifier/peft_encoder/config.py` | LoRA 전용값은 PEFT adapter config로 분리 |
| `methods/adaptation/lora_classifier/training/modeling.py` | `methods/adaptation/text_classifier/peft_encoder/modeling.py` | text classifier + encoder adapter composition |
| `methods/adaptation/lora_classifier/training/*` | `methods/adaptation/text_classifier/peft_encoder/training/*` | local training primitive |
| `methods/adaptation/lora_classifier/update/*` | `methods/adaptation/text_classifier/peft_encoder/update/*` | local update payload/materialization |
| `methods/adaptation/lora_classifier/aggregation/fedavg.py` | `methods/adaptation/text_classifier/aggregation/peft_encoder_fedavg_projection.py` | FedAvg algorithm이 아니라 PEFT encoder projection |
| `methods/adaptation/lora_classifier/aggregation/partitioned_delta_average.py` | `methods/adaptation/text_classifier/aggregation/peft_encoder_partitioned_projection.py` | sigma/psi projection |
| `methods/adaptation/lora_classifier/aggregation/state_projection.py` | `methods/adaptation/text_classifier/aggregation/peft_encoder_state_projection.py` | canonical tensor state projection |
| `methods/adaptation/lora_classifier/federated_ssl/partitioned_*.py` | `methods/adaptation/text_classifier/peft_encoder/federated_ssl/partitioned/*.py` | method-neutral partitioned primitive |
| `methods/adaptation/classifier_head/*` | `methods/adaptation/classification/feature_head/*` | modality-independent feature-head variant |
| `methods/adaptation/classifier_head/aggregation/fedavg.py` | `methods/adaptation/classification/aggregation/feature_head_fedavg_projection.py` | feature head projection |
| `methods/adaptation/lora/lora_adapter.py` | `methods/adaptation/peft_adapters/lora/builder.py` | LoRA mechanism 구현 |
| `methods/adaptation/peft/base.py` | `methods/adaptation/peft_adapters/base.py` | PEFT mechanism interface |
| `methods/adaptation/peft/registry.py` | `methods/adaptation/peft_adapters/registry.py` | mechanism lookup |

## 단계별 계획

### 0단계: 문서와 guard 고정

- 이 문서를 기준으로 migration scope를 고정한다.
- `lora_classifier` 이름을 contract/runtime 표면에서 바로 제거하지 않는다고 명시한다.
- architecture guard 후보를 정한다.
  `methods/adaptation/text_classifier/**`는 `methods.federated_ssl.fedmatch`를
  import하지 않는다.
- `methods/adaptation/text_classifier/**` 새 내부 코드는
  `methods.adaptation.lora_classifier`를 import하지 않는다.
- `methods/adaptation/text_classifier/aggregation/**`는 weighted average policy를
  직접 구현하지 않고 generic FedAvg core projection만 소유한다.
- `methods/adaptation/classification/**`는 `methods.adaptation.text_classifier`와
  legacy `methods.adaptation.classifier_head`를 import하지 않는다.
- `methods/adaptation/peft_adapters/**`는 classifier label, task payload,
  update payload를 import하지 않는다.
- 검증은 docs diff, `git diff --check`, architecture test 후보 확인으로 닫는다.

### 1단계: 목표 폴더와 README만 생성

- `classification/`, `classification/feature_head/`,
  `classification/aggregation/`, `text_classifier/`, `text_classifier/peft_encoder/`,
  `text_classifier/aggregation/`,
  `peft_adapters/` README를 먼저 만든다.
- 코드 import는 아직 바꾸지 않는다.
- README에는 owner, allowed dependency, 금지 dependency를 짧게 적는다.

### 2단계: partitioned primitive in-place 중립화

- 아직 파일을 이동하지 않는다.
- `lora_classifier/federated_ssl/partitioned_training_loop.py` 같은 현재 파일에서
  FedMatch 타입, partition 상수, objective 함수를 직접 import하지 않게 만든다.
- `PartitionedLocalObjective`, `PartitionedObjectiveResult`, `PartitionRouting`
  같은 method-neutral seam을 먼저 만든다.
- FedMatch 전용 objective와 partition routing은
  `methods/federated_ssl/fedmatch/`에서 callable/config로 만들어 주입한다.

### 3단계: PEFT encoder text classifier core 이동

- `lora_classifier/config.py`, `training/*`, `update/*`,
  `evaluation.py`, `initial_state.py`, `server_preflight.py`,
  `runtime_compatibility.py`를 `text_classifier/peft_encoder/` 아래로 이동한다.
- `lora_classifier/`는 기존 contract 이름을 받는 compatibility shim으로 축소한다.
- LoRA 전용 target module, PEFT adapter 생성 로직은 `peft_adapters/lora/`로 내린다.
- DoRA는 구현하지 않고, adapter mechanism seam이 LoRA 이름에 잠기지 않았는지만
  테스트한다.
- `initial_query_ssl_algorithm_state` 입력과 `query_ssl_algorithm_state` 결과 필드는
  새 경로에서도 유지한다.

상태: 완료. `config.py`, `evaluation.py`, `initial_state.py`,
`runtime_compatibility.py`, `server_preflight.py`, `training_backend.py`,
`training/*`, `update/*`, `aggregation/materialization.py`를
`text_classifier/peft_encoder/`로 이동했다. 기존 `lora_classifier` 경로는
direct-file compatibility shim으로 남겼고, 새 내부 코드는 legacy
`methods.adaptation.lora_classifier`를 import하지 않는다. PEFT mechanism protocol,
registry, LoRA builder도 `peft_adapters/`로 이동했다. 기존 `peft/`, `lora/`
compatibility package는 내부 import가 끊긴 뒤 제거했다.

### 4단계: 중립화된 partitioned primitive 이동

- 2단계에서 FedMatch 직접 의존이 제거된 partitioned 파일만
  `text_classifier/peft_encoder/federated_ssl/partitioned/`로 이동한다.
- 파일명은 `training_loop`, `model_builder`, `trainable_model`, `sparse_sync`,
  `budget`처럼 primitive 이름을 쓴다.
- 기존 direct file path는 compatibility shim으로 남기고 제거 조건을 README에 적는다.

상태: 완료. `partition_sparse_sync.py`, `partitioned_budget.py`,
`partitioned_model_builder.py`, `partitioned_trainable_model.py`,
`partitioned_training_loop.py`를 `text_classifier/peft_encoder/federated_ssl/partitioned/`
아래의 `sparse_sync.py`, `budget.py`, `model_builder.py`, `trainable_model.py`,
`training_loop.py`로 이동했다. 기존 `lora_classifier/federated_ssl` direct path는
named-symbol compatibility shim으로만 남겼다.

### 5단계: feature-head variant 이동

- `classifier_head/`를 `classification/feature_head/`로 이동한다.
- `classifier_head/`는 compatibility shim으로 남기고 internal import를 새 경로로
  바꾼 뒤 제거한다.
- feature-head aggregation 파일은 FedAvg algorithm이 아니라
  feature-head state projection이라는 이름으로 바꾼다.

상태: 완료 후 보정. `bootstrap.py`, `scoring.py`는
`classification/feature_head/`로 이동했고, `aggregation/fedavg.py`는
`classification/aggregation/feature_head_fedavg_projection.py`로 이동했다.
이전 단계의 `text_classifier/feature_head`와 기존 `classifier_head` direct path는
내부 import가 끊긴 뒤 제거했다. `scoring_registry.py`는 깊은
`*/scoring.py` 모듈을 스캔하므로 `classifier_head_logits` 등록이 legacy shim에
의존하지 않는다.

### 6단계: aggregation projection rename

- generic FedAvg core는 이미 `methods/federated/aggregation/`에 있다.
- `methods/adaptation/**/aggregation/fedavg.py`가 generic FedAvg core를 재구현하지
  않는지 확인한다.
- adaptation family는 canonical tensor state 추출, aggregation input 생성,
  aggregation result materialization만 소유한다.
- 관련 테스트 이름도 `fedavg` 구현 테스트가 아니라 projection/materialization 테스트로
  바꾼다.

상태: 완료. PEFT encoder aggregation source of truth는
`text_classifier/aggregation/peft_encoder_fedavg_projection.py`,
`peft_encoder_partitioned_projection.py`, `peft_encoder_state_projection.py`,
`peft_encoder_partitioned_state.py`로 이동했다.
`lora_classifier/aggregation/fedavg.py`, `partitioned_delta_average.py`,
`state_projection.py`, `partitioned_state.py`, `base_state_snapshot.py`는 direct
compatibility shim으로만 남겼다. Generic weighted-average 산술은 계속
`methods/federated/aggregation/fedavg/`가 소유한다.

### 7단계: legacy import 제거와 contract v2 검토

- internal import가 모두 새 경로를 쓰면 shim allowlist를 줄인다.
- 기존 artifact와 verifier가 안정적으로 통과한 뒤에만 shared/runtime contract 이름
  변경을 검토한다.
- 후보 이름은 `text_classifier_peft_update`,
  `text_classifier_feature_head_update`처럼 task + adaptation variant 조합이다.
- `adapter_kind=lora_classifier`와 report field rename은 producer, consumer,
  verifier, docs를 같이 바꿀 수 있을 때 v2 migration으로 닫는다.

상태: 진행 중. 내부 코드는 legacy `lora_classifier/aggregation`,
`lora_classifier/update`, `lora_classifier/config.py`, `lora_classifier/training/*`,
`lora_classifier/training_backend.py`, `lora_classifier/initial_state.py`,
`lora_classifier/evaluation.py` 경로를 직접 import하지 않는다. 해당 경로들은
compatibility shim으로만 남긴다. `helper_provider.py`,
`method_owned_training.py`, `peer_predictions.py`, `server_update_policy.py`,
`supervised_seed_step.py`, `partitioned_objective_training.py`도
`text_classifier/peft_encoder/federated_ssl/`로 이동했고 legacy path는 shim으로만
남겼다. `partitioned_objective_training.py`는 method-neutral runtime plan을 받아
실행하고, FedMatch wrapper가 runtime plan을 생성해 주입한다. 다음 보정 대상은
shared contract v2 전까지 남길 `lora_classifier` compatibility surface의 제거
조건이다.
FedMatch scenario, local supervision, sigma/psi partition routing, upload partition,
objective 생성, `psi_factor` 해석은
`methods/federated_ssl/fedmatch/partitioned_runtime_plan.py`로 분리했다.
`peft/`, `lora/`, `classifier_head/`, `text_classifier/feature_head`,
`text_classifier/aggregation/feature_head_fedavg_projection.py` legacy shim은 제거했고
architecture guard는 해당 경로가 다시 생기지 않게 막는다.
`lora_classifier` package는 `docs/contracts/legacy_contract_ledger.md`와
code-adjacent README에 유지 이유와 제거 조건을 기록했고, architecture guard는 새
business logic 파일 추가를 금지한다. 남은 `lora_classifier` runtime/config/report
용어는 `docs/contracts/lora_classifier_v1_terminology_audit.md` 기준으로 v1 contract
표면과 구현 owner 표면을 구분한다.

## 호환성 정책

- 기존 `lora_classifier` run artifact, manifest, report verifier는 즉시 깨지면 안 된다.
- compatibility shim은 business rule을 소유하지 않고 새 경로 import와 deprecation
  message만 허용한다.
- `__init__.py` re-export는 금지한다. 기존 direct file path shim만 allowlist로 허용한다.
- 모든 internal import가 새 경로를 쓰고, shared contract v2 또는 명시적 legacy ledger가
  준비되면 shim을 제거한다.
- `.py` package-level re-export는 만들지 않는다. 필요한 caller만 direct-file import를
  새 경로로 바꾼다.

## 검증 기준

각 단계는 가능한 작은 PR/commit으로 닫고, 다음 중 해당되는 검증을 실행한다.

```bash
uv run pytest tests/architecture/test_layer_dependencies.py
uv run pytest tests/unit/test_partitioned_trainable_model.py tests/unit/test_methods_fedmatch_lora_partitioned_training.py tests/unit/test_methods_lora_classifier_aggregation.py tests/unit/test_methods_server_update_materialization.py shared/tests/unit/test_adapter_contracts.py tests/unit/test_fl_ssl_report_verification.py
uv run ruff check methods tests scripts shared
uv run ruff format --check methods tests scripts shared
git diff --check
```

full-budget main run은 기본 검증이 아니다. 폴더 이동과 compatibility 검증은 unit,
integration, reduced smoke로 먼저 닫고, main run은 별도 실험 concern으로 분리한다.

## 비목표

지금은 DoRA 구현, shared payload schema rename, FedMatch 수식 튜닝, full-budget
성능 개선을 하지 않는다. `agent`나 `main_server`에 method-specific 파일도 추가하지
않고, 실제 field 의미와 payload 의미는 shared contract와 code-adjacent README가 계속
소유한다.
