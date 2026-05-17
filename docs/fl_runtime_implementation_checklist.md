# FL Runtime Implementation Checklist

이 문서는 FL 트랙의 현재 구현 상태와 다음 작업만 남기는 짧은 체크리스트다.
연구 순서는 `docs/project_execution_plan.md`, 코드 경계는
`docs/architecture/system-overview.md`, 전략 축은 `docs/strategy_surface_map.md`를
source of truth로 본다.

## 현재 상태

- `main_server`는 round lifecycle, update ingest, aggregation, publication
  scaffold를 갖고 있다.
- `agent`는 active round fetch, local training, pseudo-label selection, update
  upload scaffold를 갖고 있다.
- `scripts/experiments/fl_ssl`는 FL SSL simulation, seed sweep, client-count
  sweep, report dump를 갖고 있다.
- 활성 FL SSL method baseline은 `fedavg_pseudo_label`이다.
- `diagonal_scale`와 `lora_classifier` adapter family의 FedAvg core/projection은
  `methods/adaptation/<family>/`에 있다.
- 공통 분류 metric 계산은 `methods/evaluation`이 소유하고, FL report는 중앙 SSL과
  같은 metric shape를 재사용한다.

## Report / Evaluation

- [x] final/initial/round/client validation에 `loss`, accuracy, macro/weighted F1,
  balanced accuracy, worst-category metric, ECE/max-ECE를 남긴다.
- [x] round progression, best macro-F1 round, best loss round, round delta를 남긴다.
- [x] client split label distribution, entropy, labeled/unlabeled count와
  split skew summary를 남긴다.
- [x] client별 validation 요약에 train/labeled/unlabeled count, accepted ratio,
  update 생성 여부, update norm 진단을 함께 남긴다.
- [x] accepted-count 기반 aggregation weight proxy, zero-update client, update norm,
  communication proxy를 남긴다.
- [x] round index와 early-stop 후보 진단을 남긴다.
- [x] validation curve, primary 기준 best round, round/client time과 payload byte
  계측 상태를 report에 남긴다.
- [x] 중앙 SSL control report와 FL SSL main comparison report를 같은 ranking으로
  합치지 않는다.
- [x] `theta` 같은 method 내부 파라미터는 기본 report에 노출하지 않는다.
- [ ] 실제 main run 산출물에서 report schema를 샘플로 고정하고 dashboard/index
  소비 필드를 확정한다.

주의: FL prototype score의 `loss`는 현재 raw score를 softmax 분포로 바꾼 NLL
proxy다. report의 `loss_kind`와 `score_distribution_kind`를 같이 읽어야 한다.

## Method Extension

- [x] method identity와 recipe metadata는 `methods/federated_ssl/<method>/`가
  소유한다.
- [x] registry는 `<method>/<method>.py` convention import를 사용한다. 같은
  convention을 따르면 새 method 추가 시 registry 목록을 수정하지 않는다.
- [x] Hydra 실행 조합은 `conf/strategy_axes/fl/*`가 소유한다.
- [x] incompatible method/profile/runtime 조합은 simulation bootstrap 전에
  compatibility validator에서 실패한다.
- [ ] FedMatch/FedLGMatch/(FL)^2 중 실제 구현할 첫 method를 확정한다.
- [ ] 확정 method의 custom round-state exchange나 server policy capability가 필요한지
  먼저 문서화한다.

새 method 기본 변경 위치:

```text
methods/federated_ssl/<method>/
conf/strategy_axes/fl/method_descriptor/<method>.yaml
conf/strategy_axes/fl/local_update_profile/*.yaml      # 필요할 때만
conf/entrypoints/fl_ssl/run_federated_simulation.yaml  # round_runtime.* leaf override
tests/unit/test_methods_federated_ssl.py
tests/unit/test_scripts_hydra_configs.py
```

`agent`나 `main_server`에 method 이름 파일을 추가해야 한다면 먼저 capability seam이
부족한지 점검한다.

## Prototype / Scoring Extension

- [x] prototype build/scoring/evidence/training input core는 `methods/prototype/*`에
  분리되어 있다.
- [x] FL validation은 scoring/evidence 결과를 공통 classification report payload로
  변환한다.
- [ ] prototype-only 또는 prototype-SSL 평가 파일이 필요하면 `scripts`에
  entrypoint/thin wrapper로 추가한다.
- [ ] 두 개 이상 실험에서 안정적으로 공유되는 prototype 평가 metric만
  `methods/evaluation`으로 승격한다.

새 prototype 평가 기본 위치:

```text
scripts/experiments/prototype_analysis/        # prototype-only 분석
scripts/experiments/fl_ssl/federated_simulation/
methods/evaluation/                            # stable metric helper만
```

## Runtime Translation

- [x] `lora_classifier` family의 state/update shape와 inline/server-owned artifact-ref
  FedAvg core를 smoke로 검증했다.
- [x] FL simulation에서 `lora_pseudo_label_v1` local profile과
  `round_runtime.adapter_family_name=lora_classifier`,
  `round_runtime.aggregation_backend_name=fedavg` leaf 조합을 compose할 수 있다.
- [x] LoRA-classifier FedAvg는 두 라운드에서
  `previous global snapshot + round aggregated delta = next global snapshot`
  수식을 테스트로 고정했다.
- [x] FL simulation inline-delta 경로도 `sim_rev_0002 = sim_rev_0001 +
  round2 applied delta` 수식을 테스트로 고정했다.
- [x] manual `Query SSL + LoRA-classifier` simulation 경로는
  `methods/ssl/algorithms/*`와 실제 PEFT LoRA/classifier local trainer를 호출한다.
- [x] client별 local optimizer step 수는 `training_task.local_epochs`,
  `training_task.batch_size`, `training_task.max_steps`,
  `query_ssl_method.unlabeled_batch_size`로 동적으로 바뀐다.
- [x] simulation adapter에서 agent-local LoRA artifact ref를 server-owned
  `aggregation_artifact::` ref로 upload/materialize하는 경로를 닫았다. 서버 direct
  submission은 여전히 server-owned ref만 수락한다.
- [x] 실제 PEFT executor 기준 LoRA 1-round smoke를 실행했다.
- [ ] winner method가 요구하는 shared family/state/update payload를 확정한다.
- [x] server accept 단계에서 update payload의 `model_id`, `base_model_revision`,
  `training_scope`, LoRA backbone/config/label schema가 active state와 맞는지
  family별 compatibility preflight로 확인한다.

## Main Comparison Gate

- [x] main split: `10 clients`, Dirichlet `alpha=0.3`, split `seed=42`,
  `50 rounds`.
- [ ] stress split: Dirichlet `alpha=0.1`, `50 rounds`. 사용자 지시로 추가
  50-round 실행은 보류한다.
- [x] materialized split: 선택된 labeled source 전체와 unlabeled source 전체를
  client에 분배하고, 실제 labeled/unlabeled ratio는 report count로 기록한다.
- [x] `client_count=1..10` sweep runner와 summary JSON을 추가했다.
- [ ] `gpu_local + mxbai` runtime에서 smoke/main/sweep 산출물을 남긴다. 현재는
  smoke, alpha=0.3 main 50-round, short ablation, short sweep 산출물까지 확인했고,
  full stress/ablation/sweep은 사용자 지시로 보류했다.
- [x] 새 FL simulation report protocol은 `embedding_adapter`와
  `local_trainer_runtime` metadata를 기록한다. 이후 논문용 산출물은 이 필드로
  `gpu_local + mxbai` 여부를 확인하고, `hash_debug`/CPU smoke 결과를 성능
  근거로 섞지 않는다.

## 완료 기준

- raw text와 개인 해석 상태는 agent-local boundary에 남는다.
- server는 round, aggregation, publication만 소유한다.
- scripts simulation은 production core를 복사하지 않고 호출한다.
- 새 method 추가 위치가 `methods/federated_ssl/<method>/`, `conf`, 필요한 capability
  adapter, test로 분명히 나뉜다.
- report 파일만으로 split, round progression, client variance, calibration,
  communication proxy를 확인할 수 있다.
