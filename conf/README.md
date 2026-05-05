# Hydra Config Layout

`conf/`는 TraceMind 실험 조합과 method/runtime 파라미터의 루트 Hydra config
공간이다. 폴더명은 사람이 탐색할 때의 의미를 우선하고, YAML `# @package`는
기존 코드가 읽는 canonical config shape를 유지한다.

예를 들어 `execution_context/dataset_asset=ourafla`는 compose 후에도
`cfg.dataset`으로 들어가고, `strategy_axes/ssl/pseudo_label_selection=...`은
`cfg.pseudo_label_algorithm`으로 들어간다.

## 현재 분류

- `entrypoints/`: `@hydra.main(config_name=...)`이 읽는 실행 시작점별 root config.
- `execution_context/`: 여러 실험이 공유하는 데이터 자산, embedding adapter, runtime 환경.
- `strategy_axes/`: 중앙 SSL과 FL SSL이 공유하거나 각 track에서 교체하는 전략 축.
- `track_presets/`: 논문 비교 track 안에서만 의미가 생기는 preset.

```text
conf/
├── entrypoints/
│   ├── central_classifier_seed/
│   ├── central_ssl_control/
│   ├── data_pipeline/
│   ├── fl_ssl/
│   ├── prototype_analysis/
│   └── prototype_pack/
├── execution_context/
│   ├── dataset_asset/
│   ├── embedding_adapter/
│   └── runtime_env/
├── strategy_axes/
│   ├── adaptation/
│   │   ├── initial_checkpoint/
│   │   ├── peft_adapter/
│   │   └── transformer_backbone/
│   ├── fl/
│   │   ├── local_update_profile/
│   │   ├── method_descriptor/
│   │   ├── round_runtime_profile/
│   │   └── shard_policy/
│   ├── prototype/
│   │   └── build_strategy/
│   └── ssl/
│       ├── augmentation/
│       ├── consistency_method/
│       └── pseudo_label_selection/
└── track_presets/
    ├── central_ssl_control/
    │   ├── lora_classifier_defaults/
    │   ├── query_source/
    │   └── training_preset/
    └── fl_ssl/
        └── simulation_preset/
```

## 이름 기준

- entrypoint config는 실행 스크립트의 시작점이다.
- execution context는 방법론 비교가 아니라 실행 재료다.
- strategy axis는 실제로 교체 가능한 계산/정책 축이다.
- track preset은 central SSL control, FL SSL처럼 비교표의 맥락 안에서 쓰는 옵션 묶음이다.

## FL SSL config contract

FL SSL simulation은 config 의미가 겹치기 쉬우므로 아래처럼 읽는다.

- `strategy_axes/fl/method_descriptor`
  - `cfg.ssl_method`로 compose된다.
  - 논문 method identity, report role, custom runtime 필요 여부를 설명한다.
  - 실제 local update 계산 조합을 단독으로 소유하지 않는다.
- `strategy_axes/fl/local_update_profile`
  - `cfg.local_update_profile`로 compose된다.
  - agent local update를 만드는 training/evidence/scoring/privacy 조합을 소유한다.
  - adapter family나 aggregation backend를 소유하지 않는다.
- `strategy_axes/fl/round_runtime_profile`
  - `cfg.round_runtime_profile`로 compose된다.
  - server round runtime의 adapter family, aggregation backend, bootstrap runtime
    knob를 소유한다.
- `strategy_axes/fl/shard_policy`
  - `cfg.shard_policy`로 compose된다.
  - non-IID client split 방식만 소유한다.
- `track_presets/fl_ssl/simulation_preset`
  - client 수, round budget, output dir 같은 track 실행 budget을 소유한다.
  - method semantics나 local update policy를 소유하지 않는다.
- `report.labeled_ratio`, `report.unlabeled_ratio`, `report.seed_count`
  - 현재 report protocol metadata다.
  - labeled/unlabeled pool split 강제와 seed sweep 실행은 별도 runner/sweep에서
    닫아야 하며, 이 필드만으로 실행이 강제됐다고 보지 않는다.

## 정리 기준

- 새 실행 시작점은 `entrypoints/<track_or_stage>/` 아래에 둔다.
- 새 reusable group은 위치만 보고도 역할이 드러나야 한다.
- preset이 1개뿐이고 해당 entrypoint 전용이면 group으로 만들지 않는다.
- YAML은 실행 조합표와 파라미터를 담고, Python 구현이나 복잡한 계산 로직은 두지 않는다.
- namespace 이동은 Hydra config test, experiment catalog, active docs 갱신을 같이 닫는다.
