# FL SSL Runtime Performance Audit

이 문서는 FL SSL runtime 최적화가 실제 run 산출물에 준 영향을 기록한다.
실행 기준과 성능 수치를 남기기 위한 operations 문서이며, payload 계약의 source of
truth는 `shared/src/contracts/adapter_contract_families/lora_classifier.py`다.

## 2026-05-23 FedMatch Reduced 비교

공통 조건:

- Method: FedMatch method-owned LoRA-classifier
- Split: shared client seed, Dirichlet alpha=0.3, clients=10, seed=42
- Budget: reduced, 5 rounds, `max_steps=20`
- Runtime: `gpu_local + mxbai`, CUDA
- Command axis: `server_update_policy=fedmatch_partitioned`,
  `update_partition_policy=partitioned`, `peer_context_policy=fixed_probe_output_knn`,
  `local_ssl_policy=fedmatch_agreement`

비교 run:

| 구분 | run directory | 관련 commit |
|---|---|---|
| 이전 | `runs/fl_ssl/fedmatch/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_shared_client_seed_seed42/clients10_rounds5/20260523T123534Z` | `587a625` 이후 timing instrumentation 상태 |
| 이후 | `runs/fl_ssl/fedmatch/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_shared_client_seed_seed42/clients10_rounds5/20260523T135042Z` | `55431f1` partitioned update artifact ref 적용 |

적용된 변경:

- agent-local shared update 사본 저장 기본 비활성화
- FedMatch partitioned update payload의 inline `partitioned_deltas` 제거
- `partitioned_deltas_artifact_ref`로 server-owned partitioned delta artifact 참조
- FedMatch partitioned 경로에서 primary LoRA/head client delta artifact 생략
- report에 `client_payload_bytes`, `client_artifact_bytes`,
  `total_update_material_bytes` 분리 기록

## 결과

| 지표 | 이전 | 이후 | 변화 |
|---|---:|---:|---:|
| 전체 run directory size | `29G` | `13G` | 약 `55%` 감소 |
| `main_server/shared_adapter_updates` | `12G` | `208K` | 사실상 제거 |
| `agents/` update 사본 | `12G` | 없음 | 제거 |
| `main_server/aggregation_artifacts` | `5.9G` | `13G` | partitioned material이 canonical artifact로 이동 |
| report `total_payload_bytes` | `8,137,748,274` | `61,509` | payload JSON 비용 제거 |
| report `total_artifact_bytes` | 미계측 | `11,979,217,047` | artifact material 별도 계측 |
| report `total_update_material_bytes` | 미계측 | `11,979,278,556` | payload + artifact 총량 계측 |
| `server_update_submit_seconds.mean` | `4.926s` | `0.003s` | submit 병목 제거 |
| `agent_repository_save_seconds.mean` | `5.032s` | 없음 | agent-local 저장 제거 |
| `round_time_seconds.mean` | `394.96s` | `335.23s` | 약 `15%` 감소 |
| 최종 macro-F1 | `0.439508` | `0.441055` | 동등 범위 |
| 최종 accuracy | `0.472889` | `0.4735` | 동등 범위 |

해석:

- 서버 submit과 shared update JSON 중복 저장 병목은 제거됐다.
- run directory 총량은 줄었지만, partitioned delta 자체는 여전히 client update마다 약
  `240MB` JSON artifact로 저장된다.
- 남은 큰 시간은 local training loop, partitioned delta JSON artifact materialization,
  round-level finalize/validation/orchestration gap이다.
- 이 변경은 학습 objective, client selection, aggregation weight, validation 기준을
  바꾸지 않았다. 결과 지표는 동등 범위로 보는 것이 맞다.

## 적용 범위

공통 적용:

- `artifact_persistence.persist_agent_local_updates=false`는 manual FixMatch+FedAvg와
  FedMatch method-owned를 포함한 FL SSL simulation 공통 저장 정책이다.
- 이로 인해 agent-local shared update 사본 저장 시간과 디스크 중복이 줄어든다.

partitioned path 전용 적용:

- `partitioned_deltas_artifact_ref`는 FedMatch처럼 partitioned server update를 쓰는
  경로에 직접 효과가 있다.
- 일반 `FixMatch + FedAvg`가 merged delta를 쓰고 partitioned delta를 만들지 않는다면,
  이 artifact-ref 최적화의 큰 효과는 직접 적용되지 않는다.

## 다음 최적화 후보

1. Round-level gap 계측
   - 현재 round time에서 client training 합을 빼면 round당 `56-71s`가 남는다.
   - `round_finalize_seconds`, `round_validation_seconds`,
     `round_peer_state_seconds`를 report에 추가해 finalize/validation/orchestration 중
     어디가 큰지 분리한다.
   - 2026-05-23 현재 코드에는 `round_timing_breakdown` metadata가 추가됐다. 다음
     reduced run부터 실제 gap breakdown을 report에서 확인한다.
2. Partitioned delta binary artifact
   - 2026-05-23 현재 코드에는 server-owned partitioned delta를 `safetensors`
     artifact로 저장하는 경로가 추가됐다.
   - 기존 reduced run 기준 `partitioned_delta.json`은 client update별 약 `240MB`였고,
     float32 raw payload는 약 `27MB`였다. 새 run에서는 partitioned artifact 총량이
     약 `12GB`에서 `1.4-2GB` 범위로 줄 가능성이 높다.
   - dtype은 float32로 유지하므로 학습 objective, aggregation, validation 의미는
     바뀌지 않는다. 단, JSON float 문자열을 다시 파싱하던 경로가 binary tensor
     materialization으로 바뀌므로 새 reduced run으로 실제 size/runtime을 기록해야 한다.

## 2026-05-23 Partitioned Delta Binary Format 변경

적용된 변경:

- `partitioned_deltas_artifact_ref`의 opaque ref 의미는 유지했다.
- server-owned partitioned update artifact만 `safetensors`로 저장한다.
- agent-local JSON debug artifact와 legacy JSON artifact ref materialization은 유지한다.
- aggregation consumer는 safetensors가 있으면 우선 읽고, 없으면 기존 JSON artifact를
  읽는다.
- FedMatch LoRA partitioned step은 step마다 `LoraClassifierPartitionDelta`를 만들지
  않고 tensor delta만 반환한다. 최종 client update용 partition materialization은 local
  training loop 종료 후 누적 delta에서 한 번만 수행한다.

별도 semantic audit 대상:

- 현재 FedMatch method-owned LoRA runtime은 Query SSL multiview dataloader를 재사용해
  weak/strong view를 만든다.
- FedMatch agreement objective에서는 weak logits로 agreement/confidence를 만들고 strong
  logits에 pseudo-label CE를 적용한다.
- 이 저장 포맷 변경은 해당 objective 의미를 바꾸지 않는다. original FedMatch에 더
  충실한 single-view 해석이 필요한지는 별도 ablation/semantic audit으로 다룬다.

## 2026-05-24 Labels-at-server smoke

목적:

- `server_only_seed + supervised_seed_step + client_unlabeled_only` 경로가 실제
  simulation에서 round open 전 server supervised seed step을 실행하고, client는
  `psi` partition-only update를 제출하는지 확인한다.
- reduced run 전에 verifier가 `partitioned_deltas_artifact_ref`만 있는
  server-owned update artifact를 PASS할 수 있는지 확인한다.

공통 조건:

- Method: FedMatch method-owned LoRA-classifier
- Scenario: `labels-at-server`
- Split: server-only seed, Dirichlet alpha=0.3, clients=2, seed=42
- Budget: smoke override, 1 round, `training_task.max_steps=1`
- Runtime: `gpu_local + mxbai`, CUDA
- Capability: `server_step_policy=supervised_seed_step`,
  `server_update_policy=fedmatch_partitioned`,
  `local_supervision_regime=client_unlabeled_only`,
  `local_ssl_policy=fedmatch_agreement`

결과:

| 구분 | 값 |
|---|---:|
| 원본 `server_batch_size=100` | CUDA OOM |
| 실행 override `server_batch_size=12` | PASS |
| GPU peak 관찰값 | 약 `6.6GB / 16GB` |
| 전체 1-round smoke | `111.95s` |
| `round_server_step_seconds` | `69.77s` |
| `round_client_execution_seconds` | `11.69s` |
| `round_finalize_publication_seconds` | `8.89s` |
| `round_validation_seconds` | `21.59s` |
| final macro-F1 | `0.736076` |

검증:

- Report:
  `runs/_smoke/fl_ssl/fedmatch/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_server_only_seed_seed42/clients2_rounds1/20260523T160803Z/reports/fl_ssl_main_comparison.report.json`
- `verify_federated_report_artifacts.py`는
  `--expect-server-owned-update-artifacts` 사용 시 primary LoRA/head artifact뿐 아니라
  `partitioned_deltas_artifact_ref` 단독 update도 server-owned artifact로 검증한다.
- 이 smoke는 `partitioned_delta.safetensors` 존재, agent-local ref 미노출, 최종
  LoRA-classifier aggregate snapshot 존재까지 PASS했다.

해석:

- 원본 FedMatch의 `server_batch_size=100`은 CIFAR/ResNet 기준 값이라
  Transformer LoRA-classifier server step에는 그대로 적용하기 어렵다.
- main fair comparison의 LoRA-classifier batch 기준과 맞추기 위해 labels-at-server
  reduced run은 원본값을 report에 보존하되
  `+ssl_method.parameter_overrides.server_batch_size=12`를 명시한다.
  이 override는 batch 메모리 단위 조정이며 labeled source, objective, round count,
  client local budget 의미를 바꾸지 않는다.
