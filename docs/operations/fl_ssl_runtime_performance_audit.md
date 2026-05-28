# FL SSL Runtime Performance Audit

이 문서는 FL SSL runtime 최적화가 실제 run 산출물에 준 영향을 기록한다.
실행 기준과 성능 수치를 남기기 위한 operations 문서이며, payload 계약의 source of
truth는 `shared/src/contracts/adapter_contract_families/lora_classifier.py`다.

## 2026-05-26 FedMatch Reduced 검증

목적:

- FedMatch method-owned `fedmatch_partitioned` 경로를 `10 clients x 5 rounds`
  reduced budget에서 실제 완주시키고, report verifier로 protocol drift를 막는다.
- main/full-budget은 이 검증 범위에 포함하지 않는다.

실행 조건:

- Method: `fl_method.composition_mode=method_owned`,
  `strategy_axes/fl/method_descriptor=fedmatch`
- Runtime axes: `update_partition_policy=partitioned`,
  `aggregation_weight_policy=uniform`, `peer_context_policy=fixed_probe_output_knn`
- Method-derived policies: `server_update_policy=fedmatch_partitioned`,
  `local_ssl_policy=fedmatch_agreement`
- Split: `shared_general_reddit_pc100_alpha03_clients10`
- Budget: `run_controls/fl_ssl/budget=reduced`, `10 clients`, `5 rounds`,
  `training_task.max_steps=20`
- Run directory:
  `runs/fl_ssl/fedmatch/fedmatch__lora_classifier__fedmatch_partitioned/labeled-szegeelim_general4_unlabeled-ourafla_reddit_labels_pc100_shared_client_seed42/clients10_rounds5/20260526T120100Z`

메모리 관련 변경:

- FedMatch helper model cache를 client 경계에서 폐기한다.
- mxbai backbone base cache도 client 경계에서 폐기하고 tokenizer cache만 유지한다.
- round 간 보존되는 peer/client partition snapshot vector는 in-memory에서 float32
  array로 압축하고, JSON artifact payload로 쓸 때만 list로 정규화한다.
- 위 변경은 FedMatch sigma/psi state 의미를 바꾸지 않고 runtime 표현과 cache
  lifecycle만 바꾼다.

검증 결과:

- historical post-run communication backfill PASS. 이 CLI는 이후 삭제됐고, 새 run은
  report 생성 경로에서 artifact communication estimate를 직접 기록한다.
- `verify_federated_report_artifacts` PASS
- Posthoc communication estimate:
  C2S `1,429,494,473 bytes`, S2C `19,496,932,570 bytes`, total
  `20,926,427,043 bytes`
- Run directory size: `3.4G`
- Initial validation: macro-F1 `0.265190`, loss `1.371803`, accuracy `0.336827`,
  ECE `0.016458`
- Final validation: macro-F1 `0.138327`, loss `2.880688`, accuracy `0.259424`,
  ECE `0.381286`

해석:

- protocol과 artifact 경로는 `fedmatch_partitioned`, `partitioned`,
  `fixed_probe_output_knn`, `fedmatch_agreement`,
  `partitioned_deltas_artifact_ref`, sparse S2C communication estimate까지 verifier로
  고정됐다.
- 성능은 초기 모델보다 악화됐다. 이 기록은 실행 경로 검증이지 FedMatch 성능
  우위의 근거가 아니다.
- `clients10_rounds30/20260526T122807Z`는 사용자 요청으로 중단한 partial main
  artifact이며 검증 산출물로 사용하지 않는다.

## 2026-05-23 FedMatch Reduced 비교

공통 조건:

- Method: FedMatch method-owned PEFT text encoder
- Split: shared client seed, Dirichlet alpha=0.3, clients=10, seed=42
- Budget: reduced, 5 rounds, `max_steps=20`
- Runtime: `gpu_local + mxbai`, CUDA
- Command axis: `update_partition_policy=partitioned`,
  `peer_context_policy=fixed_probe_output_knn`
- Method-derived policies: `server_update_policy=fedmatch_partitioned`,
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

- Method: FedMatch method-owned PEFT text encoder
- Scenario: `labels-at-server`
- Split: server-only seed, Dirichlet alpha=0.3, clients=2, seed=42
- Budget: smoke override, 1 round, `training_task.max_steps=1`
- Runtime: `gpu_local + mxbai`, CUDA
- Capability: `server_step_policy=supervised_seed_step`,
  `local_supervision_regime=client_unlabeled_only`
- Method-derived policies: `server_update_policy=fedmatch_partitioned`,
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

## 2026-05-24 Labels-at-server reduced

조건:

- Method: FedMatch method-owned LoRA-classifier
- Scenario: `labels-at-server`
- Split: server-only seed, Dirichlet alpha=0.3, clients=10, seed=42
- Budget: reduced, 5 rounds, `training_task.max_steps=20`
- Runtime: `gpu_local + mxbai`, CUDA
- Override: `+ssl_method.parameter_overrides.server_batch_size=12`

검증:

- Report:
  `runs/fl_ssl/fedmatch/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_server_only_seed_seed42/clients10_rounds5/20260523T161751Z/reports/fl_ssl_main_comparison.report.json`
- `verify_federated_report_artifacts.py` PASS:
  completed rounds 5, client count 10, server-only seed split, FedMatch descriptor,
  server-owned partitioned update artifact, agent-local ref 미노출, 최종
  LoRA-classifier aggregate snapshot 존재를 확인했다.

결과:

| 지표 | 값 |
|---|---:|
| final macro-F1 | `0.736615` |
| final accuracy | `0.729893` |
| worst-client macro-F1 | `0.324624` |
| total client updates | `50` |
| total candidates | `202,775` |
| total accepted | `163,780` |
| acceptance ratio | `0.8077` |
| report `total_payload_bytes` | `58,984` |
| report `total_artifact_bytes` | `714,708,000` |
| report `total_update_material_bytes` | `714,766,984` |
| run directory size | `2.6G` |
| `main_server/aggregation_artifacts` | `2.6G` |
| `main_server/shared_adapter_updates` | `208K` |
| partitioned safetensors count | `50` |

Round timing:

| timing key | mean | max |
|---|---:|---:|
| `round_total_seconds` | `220.361s` | `233.600s` |
| `round_server_step_seconds` | `72.242s` | `73.799s` |
| `round_client_execution_seconds` | `109.082s` | `115.007s` |
| `round_finalize_publication_seconds` | `17.109s` | `23.395s` |
| `round_validation_seconds` | `21.926s` | `22.442s` |

Client timing:

| timing key | mean | max |
|---|---:|---:|
| `local_training_total_seconds` | `10.904s` | `13.797s` |
| `core_training_loop_seconds` | `5.104s` | `5.486s` |
| `core_pseudo_label_diagnostics_seconds` | `2.078s` | `2.418s` |
| `adapter_base_materialization_seconds` | `1.640s` | `2.714s` |
| `core_peer_snapshot_build_seconds` | `0.762s` | `0.838s` |
| `core_model_prepare_seconds` | `0.590s` | `2.146s` |
| `core_delta_extract_seconds` | `0.344s` | `0.455s` |
| `server_update_submit_seconds` | `0.002s` | `0.011s` |

해석:

- `server_batch_size=12`는 10-client reduced에서도 OOM 없이 안정적으로 통과했다.
- round 평균에서 server supervised seed step이 약 33%, client execution이 약 50%를
  차지한다. labels-at-server reduced의 다음 병목 후보는 server seed step 반복 비용,
  client sequential execution, validation/materialization이다.
- partitioned update는 `safetensors` 50개로 저장됐고, shared update JSON은 208KB로
  유지됐다. partitioned-only verifier 보강 후 `--expect-server-owned-update-artifacts`
  검증까지 PASS한다.

## 2026-05-24 Manual FixMatch+FedAvg materialized selector reduced

목적:

- `strategy_axes/fl/materialized_split` selector가 긴
  `fl_data.split_manifest=...` override 없이 source pair, labeled budget, shard
  policy, manifest path를 함께 고정하는지 확인한다.
- FedMatch 전용 경로가 아니라 manual `FixMatch + FedAvg + LoRA-classifier`에서도
  공통 FL SSL runtime 최적화가 적용되는지 확인한다.

조건:

- Method composition: manual `fixmatch_usb_v1 + lora_classifier + fedavg`
- Selector:
  `strategy_axes/fl/materialized_split=shared_general_reddit_pc100_alpha03_clients10`
- Source pair: labeled `szegeelim_general4`, unlabeled/validation/test
  `ourafla_reddit`
- Labeled budget: `100/class`, shared-client exposure
- Split: Dirichlet alpha=0.3, clients=10, seed=42
- Budget: reduced, 5 rounds, `training_task.max_steps=20`
- Runtime: `gpu_local + mxbai`, CUDA

검증:

- 첫 실행은 manifest 검증에서 중단됐다. 원인은 selector가 manifest path와
  source pair만 고정하고 `shard_policy` 기본값은 `label_dominant`로 남긴 것이다.
- selector YAML에 `shard_policy=dirichlet_label_skew(alpha=0.3)`,
  `bootstrap_ratio=0.2`, `labeled_exposure_policy=shared_client_seed`를 추가한 뒤
  compose와 reduced run이 통과했다.
- Report:
  `runs/fl_ssl/manual_baselines/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_shared_client_seed_seed42/clients10_rounds5/20260523T172345Z/reports/fl_ssl_main_comparison.report.json`

결과:

| 지표 | 값 |
|---|---:|
| final macro-F1 | `0.686146` |
| final accuracy | `0.712760` |
| worst-client macro-F1 | `0.319268` |
| total client updates | `50` |
| total candidates | `202,775` |
| total accepted | `750` |
| acceptance ratio | `0.0037` |
| report `total_payload_bytes` | `66,737` |
| report `total_artifact_bytes` | `5,242,689,449` |
| report `total_update_material_bytes` | `5,242,756,186` |

Round timing:

| timing key | mean | max |
|---|---:|---:|
| `round_total_seconds` | `189.822s` | `194.079s` |
| `round_client_execution_seconds` | `148.588s` | `152.481s` |
| `round_finalize_publication_seconds` | `19.457s` | `19.922s` |
| `round_validation_seconds` | `21.774s` | `21.992s` |

Client timing:

| timing key | mean | max |
|---|---:|---:|
| `local_training_total_seconds` | `14.855s` | `16.056s` |
| `core_training_loop_seconds` | `8.062s` | `8.526s` |
| `core_delta_materialization_seconds` | `2.377s` | `2.480s` |
| `core_pseudo_label_diagnostics_seconds` | `2.044s` | `2.410s` |
| `adapter_base_materialization_seconds` | `1.340s` | `1.742s` |
| `core_model_prepare_seconds` | `0.557s` | `0.742s` |
| `core_delta_extract_seconds` | `0.293s` | `0.345s` |
| `server_update_submit_seconds` | `0.002s` | `0.011s` |

해석:

- selector는 source pair와 manifest metadata를 정상 기록했다. report의
  `fl_data_source.source_selection`은 labeled `szegeelim_general4`,
  unlabeled/validation/test `ourafla_reddit`이고, `labeled_policy`는
  `count_per_class=100`이다.
- 공통 최적화는 manual baseline에도 적용됐다. agent-local update 중복 저장은
  꺼져 있고, diagnostic view와 timing breakdown도 동일 report에 남았다.
- 이 조합은 FixMatch acceptance가 낮다. 5-round 전체 acceptance ratio는 `0.0037`로,
  초반 round는 accepted pseudo-label이 거의 없고 후반에 일부만 생긴다. 따라서 이
  결과는 selector/runtime 검증으로 보고, SSL 효과 비교에는 threshold/budget/source
  ablation이 필요하다.
- manual FedAvg merged delta는 FedMatch partitioned safetensors 경로와 다르므로
  partitioned artifact 최적화 효과는 직접 적용되지 않는다. 남은 큰 병목은
  client sequential execution, client별 training loop, merged delta materialization,
  validation/finalization이다.

## 2026-05-24 Manual Merged Delta Binary Format 변경

적용된 변경:

- manual Query SSL/FedAvg 계열의 server-owned merged PEFT/head client delta artifact를
  JSON 대신 `safetensors`로 저장한다.
- `peft_adapter_delta_artifact_ref`와 `classifier_head_delta_artifact_ref`의 opaque ref 의미는
  유지했다. 같은 ref에서 `.safetensors`를 우선 읽고, 파일이 없으면 기존 JSON artifact를
  읽는다.
- FedMatch partitioned 경로는 이미 `partitioned_deltas_artifact_ref`를 `safetensors`로
  저장하고 있었으므로 이 변경의 직접 대상은 FixMatch/FlexMatch/FreeMatch/PseudoLabel
  + FedAvg 같은 merged delta 경로다.
- run-scoped `RuntimeResourceCache`를 manual Query SSL PEFT local training에도 연결했다.
  method-owned 경로와 동일하게 tokenizer와 backbone base를 cache하고, client별 학습
  모델은 cached base를 `deepcopy`한 뒤 새 PEFT/head를 붙인다.
- `core_model_prepare_seconds` 내부를 `core_seed_seconds`,
  `core_model_build_seconds`, `core_base_parameter_load_seconds`로 추가 계측한다.

의미 보존:

- dtype은 기존 tensor delta와 같은 float32 저장 경로다.
- aggregation consumer는 materialized `dict[str, list[float]]`로 복원한 뒤 기존 FedAvg
  계산을 그대로 수행한다.
- diagnostic forward 재사용, tokenization cache, mixed precision, client 병렬화는 적용하지
  않았다. 이들은 결과/상태 격리 또는 실행환경 변수 영향이 있어 별도 ablation 전까지
  보류한다.

기대 효과와 다음 측정:

- 직전 manual reduced run에서 `total_artifact_bytes`는 약 `5.24GB`,
  `core_delta_materialization_seconds.mean`은 `2.377s/client`였다.
- 새 포맷은 JSON float 문자열 직렬화와 파싱을 제거하므로 artifact size와
  `core_delta_materialization_seconds`가 크게 줄어야 한다.
- 실제 개선 폭은 다음 manual reduced run에서 같은 selector
  `shared_general_reddit_pc100_alpha03_clients10`, 10 clients, 5 rounds,
  `max_steps=20` 조건으로 기록한다.

## 2026-05-24 Manual Merged Delta Binary Format Reduced 재실행

조건:

- Method composition: manual `fixmatch_usb_v1 + lora_classifier + fedavg`
- Selector:
  `strategy_axes/fl/materialized_split=shared_general_reddit_pc100_alpha03_clients10`
- Source pair: labeled `szegeelim_general4`, unlabeled/validation/test
  `ourafla_reddit`
- Labeled budget: `100/class`, shared-client exposure
- Split: Dirichlet alpha=0.3, clients=10, seed=42
- Budget: reduced, 5 rounds, `training_task.max_steps=20`
- Runtime: `gpu_local + mxbai`, CUDA

비교 run:

| 구분 | run directory | 저장 포맷 |
|---|---|---|
| 이전 | `runs/fl_ssl/manual_baselines/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_shared_client_seed_seed42/clients10_rounds5/20260523T172345Z` | merged LoRA/head delta JSON |
| 이후 | `runs/fl_ssl/manual_baselines/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_shared_client_seed_seed42/clients10_rounds5/20260523T181111Z` | merged LoRA/head delta `safetensors` |

결과:

| 지표 | 이전 | 이후 | 변화 |
|---|---:|---:|---:|
| final macro-F1 | `0.686146` | `0.685966` | `-0.000180` |
| final accuracy | `0.712760` | `0.712760` | 동일 |
| total accepted | `750` | `743` | `-7` |
| report `total_artifact_bytes` | `5,242,689,449` | `714,566,000` | `-86.37%` |
| report `total_update_material_bytes` | `5,242,756,186` | `714,632,745` | `-86.37%` |
| run directory size | `5.9G` | `1.7G` | 약 `71%` 감소 |
| `main_server/aggregation_artifacts` | `5.9G` | `1.7G` | 약 `71%` 감소 |
| client update artifact files | `100 JSON` | `100 safetensors` | binary tensor 전환 |
| `round_time_seconds.mean` | `189.822s` | `159.416s` | `-16.02%` |
| `round_client_execution_seconds.mean` | `148.588s` | `125.036s` | `-15.85%` |
| `round_finalize_publication_seconds.mean` | `19.457s` | `12.621s` | `-35.13%` |
| `local_training_total_seconds.mean` | `14.855s` | `12.500s` | `-15.85%` |
| `core_delta_materialization_seconds.mean` | `2.377s` | `0.169s` | `-92.89%` |
| `core_model_prepare_seconds.mean` | `0.557s` | `0.421s` | `-24.50%` |
| `core_training_loop_seconds.mean` | `8.062s` | `8.074s` | `+0.15%` |
| `core_pseudo_label_diagnostics_seconds.mean` | `2.044s` | `2.051s` | `+0.33%` |

새 계측:

| timing key | mean | max |
|---|---:|---:|
| `core_model_build_seconds` | `0.332s` | `0.531s` |
| `core_base_parameter_load_seconds` | `0.088s` | `0.124s` |
| `core_seed_seconds` | `0.00035s` | `0.00082s` |

해석:

- 성능 개선의 주 원인은 merged LoRA/head delta를 JSON float 문자열로 직렬화하지 않고
  `safetensors` binary tensor artifact로 저장한 것이다.
- aggregator의 입력 의미는 그대로다. 새 consumer는 `.safetensors`를 읽어 기존과 같은
  `dict[str, list[float]]` materialized delta로 복원한 뒤 기존 FedAvg 계산을 수행한다.
- `core_training_loop_seconds`와 pseudo-label diagnostics는 거의 변하지 않았다. 따라서
  이번 개선은 학습 objective 속도 개선이 아니라 client update materialization과
  publication/finalization IO 감소다.
- final macro-F1과 accuracy는 동등 범위다. accepted pseudo-label 수 차이 `-7`은
  stochastic/local ordering 수준으로 보며, 저장 포맷 변경이 aggregation 의미를 바꾼
  신호는 아니다.
- manual runtime cache 연결로 `core_model_prepare_seconds`도 일부 줄었다. 다만 새
  세부 계측상 model build 평균 `0.332s`, base parameter load 평균 `0.088s`라서
  현재 주요 병목은 여전히 `core_training_loop_seconds`, pseudo-label diagnostics,
  validation, sequential client execution이다.

## 2026-05-24 Round Base Snapshot Cache Reduced 재실행

적용된 변경:

- simulation runtime resource로 `RoundBaseSnapshotCache`를 추가했다.
- round 시작 시 cache를 clear하고, 같은 round 안에서 동일 active global base state
  materialization을 공유한다.
- cache 자체는 family/method 의미를 모른다. LoRA-classifier는 첫 integration으로
  `adapter_kind`, `model_revision`, state artifact refs, schema version을 key로 사용한다.
- manual Query SSL/FedAvg 경로와 method-owned FedMatch 경로 모두 같은
  `round_base_snapshot_cache`를 통과한다.
- cache 대상은 materialized global base snapshot뿐이다. `nn.Module`, PEFT model,
  optimizer, dataloader, pseudo-label 결과, client-local state는 cache하지 않는다.

비교 run:

| 구분 | run directory | 적용 상태 |
|---|---|---|
| 이전 | `runs/fl_ssl/manual_baselines/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_shared_client_seed_seed42/clients10_rounds5/20260523T181111Z` | merged delta `safetensors`, base materialization client별 반복 |
| 이후 | `runs/fl_ssl/manual_baselines/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_shared_client_seed_seed42/clients10_rounds5/20260523T184942Z` | merged delta `safetensors`, round base snapshot cache |

결과:

| 지표 | 이전 | 이후 | 변화 |
|---|---:|---:|---:|
| final macro-F1 | `0.685966` | `0.685966` | 동일 |
| final accuracy | `0.712760` | `0.712760` | 동일 |
| total accepted | `743` | `743` | 동일 |
| report `total_artifact_bytes` | `714,566,000` | `714,566,000` | 동일 |
| `round_time_seconds.mean` | `159.416s` | `146.235s` | `-8.27%` |
| 5-round total time estimate | `797.082s` | `731.177s` | `-65.905s` |
| `round_client_execution_seconds.mean` | `125.036s` | `112.059s` | `-10.38%` |
| `local_training_total_seconds.mean` | `12.500s` | `11.202s` | `-10.38%` |
| `adapter_base_materialization_seconds.mean` | `1.321s` | `0.125s` | `-90.55%` |
| `adapter_base_materialization_seconds.total` | `66.049s` | `6.244s` | `-59.806s` |
| `core_model_prepare_seconds.mean` | `0.421s` | `0.343s` | `-18.55%` |
| `core_training_loop_seconds.total` | `403.696s` | `402.644s` | `-1.052s` |
| `core_pseudo_label_diagnostics_seconds.total` | `102.533s` | `102.742s` | `+0.209s` |
| `round_validation_seconds.total` | `108.790s` | `108.795s` | 동일 범위 |

해석:

- cache 적용 전 예상한 병목인 `adapter_base_materialization_seconds`가 총
  `66.049s`에서 `6.244s`로 줄었다. 5 rounds x 10 clients 구조에서 round당 첫
  materialization만 비용을 내고 나머지는 cache hit가 된 결과다.
- 5-round reduced 총 round 시간은 약 `65.9s` 줄었다. 저장 포맷 전환 이후 남은
  병목 중 base materialization 반복 비용은 대부분 제거됐다.
- final macro-F1, accuracy, total accepted가 완전히 동일하므로 이 변경은 학습 의미를
  바꾸지 않는 runtime materialization 최적화로 볼 수 있다.
- 남은 큰 병목은 `core_training_loop_seconds` 약 `402.6s`, pseudo-label diagnostics
  약 `102.7s`, validation 약 `108.8s`다. base materialization은 더 이상 1순위 병목이
  아니다.

## 2026-05-24 Tokenization/Transfer/Clip Reduced 재실행

적용된 변경:

- FL SSL local runtime의 `RuntimeResourceCache`에 run-local text tokenization cache를
  추가했다.
- labeled train/selection loader, weak unlabeled loader, multiview unlabeled loader,
  final pseudo-label diagnostics loader가 같은 tokenizer namespace와 text 기준 cache를
  공유한다.
- CUDA 환경에서 DataLoader `pin_memory=True`와 tensor 이동
  `non_blocking=True`를 사용한다.
- gradient clipping 대상은 전체 parameter iterator가 아니라
  `requires_grad=True` trainable parameter tuple로 제한한다.
- 정보 로그와 round별 diagnostic 평가 빈도는 유지했다.

비교 run:

| 구분 | run directory | 적용 상태 |
|---|---|---|
| 이전 | `runs/fl_ssl/manual_baselines/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_shared_client_seed_seed42/clients10_rounds5/20260523T184942Z` | merged delta `safetensors`, round base snapshot cache |
| 이후 | `runs/fl_ssl/manual_baselines/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_shared_client_seed_seed42/clients10_rounds5/20260523T192005Z` | tokenization cache, pinned/non-blocking transfer, trainable-only grad clipping 추가 |

결과:

| 지표 | 이전 | 이후 | 변화 |
|---|---:|---:|---:|
| final macro-F1 | `0.685966` | `0.685966` | 동일 |
| final accuracy | `0.712760` | `0.712760` | 동일 |
| final loss | `0.685471` | `0.685471` | 동일 |
| total accepted | `743` | `743` | 동일 |
| `round_time_seconds.mean` | `146.235s` | `143.398s` | `-1.94%` |
| 5-round total time estimate | `731.177s` | `716.992s` | `-14.186s` |
| `client_train_time_seconds.mean` | `11.204s` | `10.948s` | `-2.28%` |
| `core_training_loop_seconds.mean` | `8.053s` | `7.834s` | `-2.71%` |
| `core_training_loop_seconds.total` | `402.644s` | `391.715s` | `-10.929s` |
| `core_pseudo_label_diagnostics_seconds.mean` | `2.055s` | `2.014s` | `-1.97%` |
| `core_pseudo_label_diagnostics_seconds.total` | `102.742s` | `100.720s` | `-2.023s` |
| `core_delta_materialization_seconds.mean` | `0.173s` | `0.167s` | `-3.44%` |
| `adapter_base_materialization_seconds.total` | `6.244s` | `6.461s` | `+0.217s` |

해석:

- metric, loss, accepted 수가 완전히 동일하므로 이번 변경은 학습 의미를 바꾸지 않는
  runtime 최적화로 볼 수 있다.
- 개선 폭은 저장 포맷 전환이나 round base cache보다 작다. 병목 대부분은 여전히 실제
  forward/backward와 diagnostic forward에 남아 있기 때문이다.
- `core_training_loop_seconds.total`이 약 `10.9s` 줄었고, 전체 5-round 합산은 약
  `14.2s` 줄었다. 이 개선은 반복 tokenization 감소, pinned/non-blocking transfer,
  trainable-only gradient clipping이 합쳐진 결과이며, 현재 report에는 세 항목별 분리
  타이밍은 없다.
- round별 diagnostic 평가는 유지했기 때문에 report 정보량은 줄지 않았다.
