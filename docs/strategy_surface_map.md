# 전략 표면 맵

이 문서는 현재 TraceMind에서 바꿀 수 있는 전략 축과 아직 metadata만 있는 축을
짧게 보여준다. 추가 절차는 `docs/contracts/algorithm_extension_guide.md`와
`docs/contracts/strategy_addition_playbook.md`를 따른다.

## 현재 기본 방향

```text
central fixed embedding + classifier seed
-> central SSL pooled/offline control
-> FL SSL non-IID main comparison
-> FL/runtime translation
```

- 중앙 SSL은 pooled/offline control이다.
- 논문 메인 비교는 `FL SSL under non-IID`에서 수행한다.
- runtime v1은 `embedding -> global classifier -> local interpretation` baseline을 우선한다.
- prototype은 bootstrap/comparison/reference artifact와 일부 local mechanism으로 유지한다.
- prototype 기반 pseudo-label/SSL은 중앙 SSL/FL SSL 비교군에 포함한다.

## 읽는 법

- `선택 위치`: 어떤 config나 runtime field가 전략 이름을 고르는가.
- `기본값`: 값이 없을 때 fallback 되는 source of truth.
- `상태`: `활성 runtime`, `중앙 control`, `simulation`, `metadata only` 중 하나.

## 실행/모델 축

| 축 | 현재 값 | 선택 위치 | 기본값 | 상태 |
|---|---|---|---|---|
| Dataset asset | `ourafla`, `cssrs`, `mental_health_kaggle` | `execution_context/dataset_asset` | entrypoint defaults | 활성 |
| Embedding adapter | `mxbai`, `hash_debug` | `execution_context/embedding_adapter` | entrypoint defaults | 활성 |
| Query split | `dataset_default`, `ourafla_ssl_*`, `szegeelim_general4_*` | `execution_context/query_split` | entrypoint defaults | 활성 |
| Query text view | `szegeelim_general4_*_nllb_v1` | `execution_context/query_view` | data pipeline entrypoint defaults | 활성 |
| Runtime env | `cpu_local`, `gpu_local`, `gpu_online`, `auto_local`, `auto_online` | `execution_context/runtime_env` | entrypoint defaults | 활성 |
| Transformer backbone | `mxbai_encoder` | `strategy_axes/adaptation/transformer_backbone` | central SSL defaults | 중앙 control |
| Initial checkpoint | `canonical_fixed_classifier_seed`, `none`, `required` | `strategy_axes/adaptation/initial_checkpoint` | Query SSL method comparison defaults to `none` | 중앙 control |

## SSL/adaptation 축

| 축 | 현재 값 | 선택 위치 | core | 상태 |
|---|---|---|---|---|
| Pseudo-label selection | `fixed_confidence_095`, `margin_threshold_v1` | `strategy_axes/ssl/pseudo_label_selection` | `methods/ssl/hooks/*` | 활성/중앙 control |
| Consistency method | `pseudolabel_usb_v1`, `fixmatch_usb_v1`, `flexmatch_usb_v1`, `freematch_usb_v1`, `adamatch_usb_v1` | `strategy_axes/ssl/consistency_method` | `methods/ssl/algorithms/*` | 중앙 control |
| SSL augmentation | `precomputed_usb_candidates_v1` | `strategy_axes/ssl/augmentation` | scripts runner | 중앙 control |
| Query classifier adaptation | supervised, bootstrap, pseudo-label, prototype SSL, FixMatch entrypoints | `conf/entrypoints/central_ssl_control/*` | `methods/adaptation/lora_classifier/*` + query data glue | 중앙 control |
| PEFT adapter | `lora`, `rslora` | `strategy_axes/adaptation/peft_adapter` | `methods/adaptation/*` | 중앙 control seam |

## Agent local runtime 축

| 축 | 현재 값 | 선택 위치 | core/runtime | 상태 |
|---|---|---|---|---|
| Training backend | `diagonal_scale_heuristic`, `lora_classifier_trainer` | `TrainingObjectiveConfig.training_backend_name` | `methods/adaptation/*` core + agent runtime adapter | 활성 runtime / FL simulation scaffold |
| Example generation | `prototype_rescore`, `weak_strong_pair` | `TrainingObjectiveConfig.example_generation_backend_name` | `methods/prototype/training_inputs/*` core + agent runtime adapter | 활성 runtime |
| Evidence backend | `prototype_similarity_evidence` | `TrainingObjectiveConfig.evidence_backend_name` | `methods/prototype/evidence/*` core + agent runtime adapter | 활성 runtime |
| Scorer backend | `prototype_similarity`, `classifier_head_logits` | `TrainingObjectiveConfig.scorer_backend_name` | `methods/prototype/scoring/*` core + agent runtime adapter | 활성 runtime |
| Score policy | `max_cosine`, `topk_mean_cosine` | `TrainingObjectiveConfig.score_policy_name` | `methods/prototype/scoring/policy_registry.py` + `score_policies/*` | 활성 runtime |
| Acceptance policy | `top1_margin_threshold`, `top1_confidence_only` | `TrainingObjectiveConfig.acceptance_policy_name` | `methods/ssl/hooks/selection.py` + agent compatibility metadata | 활성 runtime |
| Privacy guard | `diagonal_scale_clip_only`, `classifier_head_clip_only`, `noop` | `TrainingObjectiveConfig.privacy_guard_name` | agent privacy service | 활성 runtime |

## FL/서버 축

| 축 | 현재 값 | 선택 위치 | core/runtime | 상태 |
|---|---|---|---|---|
| Shard policy | `label_dominant`, `dirichlet_alpha03`, `dirichlet_alpha01` | `strategy_axes/fl/shard_policy` | `methods/federated/shard_policy/*` | simulation |
| FL SSL method spec | `fedavg_pseudo_label` | `strategy_axes/fl/method_descriptor` | `methods/federated_ssl/*`, simulation adapter | simulation baseline |
| FL local-update profile | `prototype_pseudo_label_v1`, `prototype_top1_confidence_v1`, `lora_pseudo_label_v1` | `strategy_axes/fl/local_update_profile` -> `cfg.local_update_profile` | agent training/evidence/scoring/privacy runtime | simulation/runtime profile |
| FL round-runtime profile | `fedavg_diagonal_scale`, `fedavg_lora_classifier` | `strategy_axes/fl/round_runtime_profile` -> `cfg.round_runtime_profile` | adapter family + aggregation runtime pairing | simulation/runtime profile |
| Aggregation backend | `fedavg` | `round_runtime.aggregation_backend_name` | reusable backend는 `methods/federated/aggregation/fedavg/*` + `methods/adaptation/<family>/fedavg.py`, `fedavg_projection.py`, method-only 변형은 `methods/federated_ssl/<method>/` + main_server generic aggregation executor | 활성 runtime |
| Adapter family | `diagonal_scale`, `classifier_head`, `lora_classifier` | `round_runtime.adapter_family_name`, model/update manifest | shared contracts, main_server generic family runtime | 활성 runtime / server aggregation scaffold |
| Update acceptance | composite round policy | main_server round service | main_server acceptance service | 활성 runtime |
| Secure update codec | `noop` | shared service/runtime wiring | `shared/src/services/secure_update_codec.py` | 활성 placeholder |

주의:

- `local_update_profile`은 local update 실행 조합 preset이고,
  `round_runtime_profile`은 server round의 adapter family와 aggregation backend 실행
  조합 preset이다. method-specific local objective와 server policy 의미는 profile이
  아니라 `methods/`의 descriptor/policy module이 소유한다.
- method identity와 local/server policy 의미는 `methods/`가 소유한다. `agent`와
  `main_server`의 backend/adapter는 선택된 core를 runtime data, artifact, contract
  payload에 연결하는 capability다.
- 논문 방법론은 `methods/federated_ssl/<method>/`를 사람이 읽는 시작점으로 둔다.
  method-only 변형은 이 폴더에 남기고, 두 개 이상 방법론에서 공유되는 aggregation,
  adapter projection, SSL hook은 축별 methods 패키지로 승격한다.

## Prototype 축

| 축 | 현재 값 | 선택 위치 | core | 상태 |
|---|---|---|---|---|
| Build strategy | `single`, `kmeans` | `strategy_axes/prototype/build_strategy` | `methods/prototype/building/*` | 활성 |
| Scoring policy | `max_cosine`, `topk_mean_cosine` | training/scoring config | `methods/prototype/scoring/*` | 활성 |
| Evidence normalization | `prototype_similarity_evidence` | training objective config | `methods/prototype/evidence/*` | 활성 |
| Training input view | `prototype_rescore`, multiview helpers | training objective config | `methods/prototype/training_inputs/*` | 활성 |

Prototype SSL 비교군은 위 prototype mechanism을 SSL objective/selection과 조합한
method다. `PrototypePack` 자체가 최종 판정기라는 뜻은 아니다.

## Metadata-only 또는 주의 축

- Prototype translation provenance는 pack/build state metadata이며 번역 실행 knob가 아니다.
- Central SSL control의 `acceptance_policy_name`은 일부 runtime compatibility field로 남아 있다.
- Multi-prototype runtime은 실험 분석에는 가능하지만 v1 active runtime 기본값은 아니다.
- `lora_classifier`는 FL simulation research path를 1차 범위로 열고, live
  `agent`/`main_server` runtime translation은 2차 범위로 둔다.
- `lora_classifier`는 기존 `classifier_head` family의 옵션이 아니다. LoRA
  adapter state와 classifier head state를 함께 표현하는 별도 family로 설계한다.
- `lora_classifier` 기본 scaffold는 `mxbai_encoder`, LoRA
  `rank=8/alpha=16/dropout=0.1/target_modules=all-linear`, canonical seed
  checkpoint, label schema, split, seed, metric을 고정한다.
- Agent `lora_classifier_trainer`는 fixed embedding-only example을 받지 않는다.
  source row metadata 또는 translated text에서 agent-local raw text를 읽고,
  shared update payload에는 raw text 없이 LoRA/classifier artifact ref와 통계만 남긴다.
  live stored-event runtime translation은 raw text 저장 경계가 정리될 때까지 2차 범위다.
- LoRA-classifier FedAvg 경로는 inline LoRA parameter delta와 classifier-head
  delta의 FedAvg shape/version을 methods-owned core로 검증한다. Server-owned
  `aggregation_artifact::` JSON artifact-ref update는 main_server가 제공한 loader로
  materialize하며, `agent-local://` ref는 upload 경로가 붙기 전까지 거부한다.
- FL simulation은 `local_update_profile=lora_pseudo_label_v1`와
  `round_runtime_profile=fedavg_lora_classifier` 조합을 compose할 수 있다. 이
  profile은 기존 `fedavg_pseudo_label` method descriptor를 유지한다. 실제 1-round
  LoRA weight 집계는 agent-local artifact upload/materialization 또는 inline
  train-step delta가 붙은 뒤 실행한다.
