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
| Dataset asset | `ourafla`, `cssrs` | `execution_context/dataset_asset` | entrypoint defaults | 활성 |
| Embedding adapter | `mxbai`, `hash_debug` | `execution_context/embedding_adapter` | entrypoint defaults | 활성 |
| Runtime env | `cpu_local`, `gpu_local`, `gpu_online`, `auto_local`, `auto_online` | `execution_context/runtime_env` | entrypoint defaults | 활성 |
| Transformer backbone | `mxbai_encoder` | `strategy_axes/adaptation/transformer_backbone` | central SSL defaults | 중앙 control |
| Initial checkpoint | `canonical_fixed_classifier_seed`, `none`, `required` | `strategy_axes/adaptation/initial_checkpoint` | central SSL defaults | 중앙 control |

## SSL/adaptation 축

| 축 | 현재 값 | 선택 위치 | core | 상태 |
|---|---|---|---|---|
| Pseudo-label selection | `fixed_confidence_095`, `margin_threshold_v1` | `strategy_axes/ssl/pseudo_label_selection` | `methods/ssl/hooks/*` | 활성/중앙 control |
| Consistency method | `fixmatch_usb_v1` | `strategy_axes/ssl/consistency_method` | `methods/ssl/fixmatch/*` | 중앙 control |
| SSL augmentation | `precomputed_usb_candidates_v1`, `backtranslation_nllb_en_de_fr_usb_v1` | `strategy_axes/ssl/augmentation` | scripts runner + agent bridge | 중앙 control |
| Query classifier adaptation | supervised, bootstrap, pseudo-label, prototype SSL, FixMatch entrypoints | `conf/entrypoints/central_ssl_control/*` | `methods/adaptation/query_classifier_adaptation/*` | 중앙 control |
| PEFT adapter | `lora`, `rslora` | `strategy_axes/adaptation/peft_adapter` | `methods/adaptation/*` | 중앙 control seam |

## Agent local runtime 축

| 축 | 현재 값 | 선택 위치 | core/runtime | 상태 |
|---|---|---|---|---|
| Training backend | `diagonal_scale_heuristic` | `TrainingObjectiveConfig.training_backend_name` | `methods/adaptation/diagonal_scale/*`, agent backend | 활성 runtime |
| Example generation | `prototype_rescore`, `weak_strong_pair` | `TrainingObjectiveConfig.example_generation_backend_name` | `methods/prototype/training_inputs/*`, agent backend | 활성 runtime |
| Evidence backend | `prototype_similarity_evidence` | `TrainingObjectiveConfig.evidence_backend_name` | `methods/prototype/evidence/*`, agent backend | 활성 runtime |
| Scorer backend | `prototype_similarity`, `classifier_head_logits` | `TrainingObjectiveConfig.scorer_backend_name` | `methods/prototype/scoring/*`, agent backend | 활성 runtime |
| Score policy | `max_cosine`, `topk_mean_cosine` | `TrainingObjectiveConfig.score_policy_name` | `methods/prototype/scoring/policies.py` | 활성 runtime |
| Acceptance policy | `top1_margin_threshold`, `top1_confidence_only` | `TrainingObjectiveConfig.acceptance_policy_name` | agent training policy | 활성 runtime |
| Privacy guard | `diagonal_scale_clip_only`, `classifier_head_clip_only`, `noop` | `TrainingObjectiveConfig.privacy_guard_name` | agent privacy service | 활성 runtime |

## FL/서버 축

| 축 | 현재 값 | 선택 위치 | core/runtime | 상태 |
|---|---|---|---|---|
| Shard policy | `label_dominant`, `dirichlet_alpha03`, `dirichlet_alpha01` | `strategy_axes/fl/shard_policy` | `methods/federated/shard_policy/*` | simulation |
| FL SSL descriptor | `fedavg_pseudo_label` | `strategy_axes/fl/method_descriptor` | `methods/federated_ssl/*` | simulation/runtime metadata |
| FL local-update profile | `prototype_pseudo_label_v1`, `prototype_top1_confidence_v1` | `strategy_axes/fl/local_update_profile` -> `cfg.local_update_profile` | agent training/evidence/scoring/privacy runtime | simulation/runtime profile |
| FL round-runtime profile | `fedavg_diagonal_scale` | `strategy_axes/fl/round_runtime_profile` -> `cfg.round_runtime_profile` | adapter family + aggregation runtime pairing | simulation/runtime profile |
| Aggregation backend | `fedavg` | `round_runtime.aggregation_backend_name` | `methods/federated/aggregation/fedavg/*`, main_server adapter | 활성 runtime |
| Adapter family | `diagonal_scale`, `classifier_head` | `round_runtime.adapter_family_name`, model/update manifest | shared contracts, main_server family wiring | 활성 runtime |
| Update acceptance | composite round policy | main_server round service | main_server acceptance service | 활성 runtime |
| Secure update codec | `noop` | shared service/runtime wiring | `shared/src/services/secure_update_codec.py` | 활성 placeholder |

주의:

- `local_update_profile`은 agent local update policy를 소유하고,
  `round_runtime_profile`은 server round의 adapter family와 aggregation backend를
  소유한다. local objective와 aggregation/backend 조합은 독립적으로 override한다.

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
- LoRA family의 FL runtime translation은 winner 확정 후 shared payload와 adapter state부터 설계한다.
