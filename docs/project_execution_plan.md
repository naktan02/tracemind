# TraceMind Execution Plan

이 문서는 현재 활성 목표와 다음 실행 판단만 유지한다. 세부 migration 기록은
`docs/notes/**` archive이고, 최종 method/runtime 구조 판단은
`docs/architecture/target-method-runtime-structure.md`를 우선한다.

## Active Goal

```text
central fixed embedding + classifier seed
-> central SSL pooled/offline control
-> FL SSL non-IID main comparison
-> FL/runtime translation
```

- 원문 텍스트와 개인 해석 상태는 agent-local boundary에 남긴다.
- 공통 개선은 shared artifact와 global trainable state로만 이동한다.
- 중앙 SSL은 pooled/offline control이며, 최종 논문 메인 랭킹이 아니다.
- 논문 메인 비교는 non-IID FL SSL이다.
- winner는 runtime/privacy 제약을 확인한 뒤 production runtime으로 translation 한다.

## Fixed Decisions

- canonical seed artifact는 `clf_2026_04_11_143138`이다.
- `WindowSummary`, `NormPack`은 활성 경로가 아니다.
- `PrototypePack`은 bootstrap/comparison/reference artifact이며 메인 판정기가 아니다.
- query-domain 적응은 현재 `PEFT text encoder + linear head` scaffold를 쓴다.
- `classifier_head`는 shared state/payload 안의 head value object 이름으로만 남긴다.
  top-level update family나 config 축 이름으로 쓰지 않는다.
- `lora_classifier`, `adapter_family_name`, `diagonal_scale`은 active 실행 축이 아니다.
- LoRA/RSLoRA는 PEFT adapter mechanism이다. 상위 config surface는 `peft_adapter`와
  `peft_text_encoder`를 기준으로 둔다.
- 현재 canonical update family는 `peft_text_encoder`다.
- 새 trainable state가 생기면 `methods/`와 `conf/`를 먼저 확장하고, `scripts`,
  `agent`, `main_server`에는 method 이름 파일을 늘리지 않는다.

## FL SSL Comparison Contract

- main split: `10 clients`, Dirichlet label-skew `alpha=0.3`, split seed `42`.
- `alpha=0.1`은 final stress/robustness 확인용이다.
- full-budget preset: `30 communication rounds`, `local_epochs=1`, `max_steps=20`.
- smoke/reduced run으로 wiring을 먼저 확인한 뒤 full-budget 비교로 올린다.
- primary metrics: `macro-F1`, `worst-client macro-F1`.
- risk metrics: `loss`, `weighted-F1`, balanced accuracy, worst-category F1,
  `ECE/max-ECE`, communication cost, per-client variance.
- smoke artifacts는 `runs/_smoke/fl_ssl`, 논문/웹 후보 artifacts는 `runs/fl_ssl`
  아래에 둔다.

## Runtime Boundaries

- `shared/`: contract, domain entity, canonical payload 해석.
- `methods/`: SSL, FL aggregation, adaptation, prototype 계산 core.
- `conf/`: Hydra 실행 조합과 파라미터.
- `scripts/`: entrypoint, sweep, report, visualization thin wrapper.
- `agent/`: agent-local training/inference runtime, private/local state.
- `main_server/`: round lifecycle, aggregation, publication, server API.
- `apps/`: API consumer/UI shell. 계약 의미나 기본값을 소유하지 않는다.

의존 방향은 `shared -> methods -> agent/main_server/scripts`다. `scripts`가 runtime을
재사용할 때는 `scripts/runtime_adapters/`에 request/context/artifact bridge만 둔다.

## Current Checkpoint

- `lora_classifier`, `text_classifier` shim, `diagonal_scale` active package/config는
  제거된 상태를 유지한다.
- FL simulation은 `payload_adapter_kind=peft_classifier`,
  `update_family_name=peft_text_encoder`, `aggregation_backend_name=fedavg`를 쓴다.
- `strategy_axes/model_architecture/peft`는 `peft_adapter` namespace를 사용한다.
- simulation model revision naming은
  `scripts/experiments/fl_ssl/federated_simulation/model_revisions.py`가 소유한다.
- PEFT supervised seed step과 final projection core는
  `methods/adaptation/peft_text_encoder/simulation_runtime/`가 소유하고,
  scripts runtime adapter는 runtime bridge만 맡는다.
- result-index와 dashboard reader는 current `peft_adapter_*`,
  `payload_adapter_kind`, `update_family_name` field를 기준으로 읽고,
  mechanism별 PEFT option은 `peft_adapter_parameters_json` snapshot으로 보존한다.

## Next Priorities

1. `scripts/report/result-index`에서 old-run reader compatibility가 active producer
   의미로 새지 않는지 마지막으로 점검한다.
2. `conf/strategy_axes` leaf가 실행 조합과 파라미터만 소유하는지 확인하고, 얇지만
   의미 있는 leaf와 중복 leaf를 구분한다.
3. active docs가 완료된 migration 계획을 current priority처럼 보이게 하지 않도록
   archive 또는 축약한다.
4. dead folder와 architecture guard를 최종 감사한다. `data`, `runs`, historical
   artifact cache는 건드리지 않는다.

## Validation Criteria

- Seed: canonical seed artifact와 report metadata가 재현 가능하다.
- Central SSL: 같은 고정 조건에서 pooled/offline control table과 output metadata가 남는다.
- FL SSL: client partition, non-IID 정도, labeled/unlabeled policy, metric, seed,
  local/round budget이 report에 남는다.
- Runtime: update base revision, aggregation, publication, artifact rebuild가 일관된다.
- Privacy: raw text는 서버로 올라가지 않고 privacy layer는 training logic과 분리된다.

## Future Decisions

아래는 현재 구조 리팩터링을 막는 결정이 아니라, 다음 연구/제품 단계에서 고를
선택지다.

- query buffer raw text retention 기본값.
- private adapter/head 도입 시점.
- secure aggregation과 DP 도입 시점.
- prototype 기반 runtime 확장 우선순위.
