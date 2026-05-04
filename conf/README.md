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
│   │   ├── client_training_profile/
│   │   ├── method_descriptor/
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

## 정리 기준

- 새 실행 시작점은 `entrypoints/<track_or_stage>/` 아래에 둔다.
- 새 reusable group은 위치만 보고도 역할이 드러나야 한다.
- preset이 1개뿐이고 해당 entrypoint 전용이면 group으로 만들지 않는다.
- YAML은 실행 조합표와 파라미터를 담고, Python 구현이나 복잡한 계산 로직은 두지 않는다.
- namespace 이동은 Hydra config test, experiment catalog, active docs 갱신을 같이 닫는다.
