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
   - 현재 `partitioned_delta.json`은 client update별 약 `240MB`다.
   - 같은 수치를 `safetensors` 또는 `npz` 같은 binary tensor artifact로 저장하면
     `core_delta_materialization_seconds`와 artifact size를 줄일 수 있다.
   - dtype/정밀도를 바꾸지 않는 방식으로 적용해야 실험 의미가 유지된다.
