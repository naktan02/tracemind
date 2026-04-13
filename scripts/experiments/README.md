# Experiments

이 디렉터리는 코어 runtime을 조합해서 연구/검증을 수행하는 entrypoint layer다.

원칙:

- 운영 후보 로직은 `shared`, `agent`, `main_server`에 둔다.
- 이 디렉터리는 그 코어를 조합하는 실험 CLI와 실험 전용 helper만 둔다.
- Hydra top-level config는 `scripts/conf/experiments/*.yaml`이 source of truth다.

전략 축 전체와 현재 override 가능 여부는
[docs/strategy_surface_map.md](../../docs/strategy_surface_map.md)를
먼저 보는 편이 빠르다.

## 직접 실행하는 entrypoint

- `prototype_strategy_experiment.py`
  - single/kmeans/dbscan prototype 전략 비교
- `prototype_threshold_sweep.py`
  - 선택된 prototype 전략 위에서 threshold policy 비교
- `run_federated_simulation.py`
  - agent/main_server 코어를 조합한 synthetic FL loop
- `train_softmax_classifier.py`
  - 고정 임베딩 위 linear classifier baseline
- `train_lora_classifier.py`
  - query-domain 적응 단계의 `frozen backbone + LoRA + classifier` canonical supervised baseline entrypoint
- `train_lora_pseudo_label_classifier.py`
  - seed labeled split과 pseudo-labeled rows를 합쳐 `pseudo-label self-training`을 실행하는 entrypoint
- `lora_classifier/`
  - query-domain LoRA scaffold의 helper 모듈
  - `runner.py`가 canonical supervised baseline runner다.
  - `query_adaptation_runner.py`는 query adaptation dataset을 baseline runner에 연결하는 wrapper다.
  - `pseudo_label_runner.py`는 seed labeled rows와 pseudo-labeled rows를 합쳐 baseline runner에 넘긴다.
  - `query_adaptation_io.py`는 agent-local adaptation dataset을 현재 JSONL 입력 shape로 export한다.
  - `query_adaptation_multiview_io.py`는 필요 시 multiview dataset을 같은 JSONL shape로 export한다.
  - export는 `source_row.query_id`를 single source of truth로 쓰고, locale은 typed provenance에서 읽는다.
  - export는 JSONL/manifest와 함께 dataset summary JSON도 남긴다.
  - `query_adaptation_runner.py`는 labeled row를 메모리에서 바로 baseline runner에 넘기고, export 산출물은 trace/audit 용도로 함께 남긴다.

## 공통 helper

- `scripts/labeled_query_rows.py`
  - 실험/프로토타입 스크립트가 공유하는 labeled row JSONL shape
- `scripts/query_buffer_selection_diagnostics.py`
  - query-buffer selection summary/trace dump 저장 helper
- `scripts/run_artifacts.py`
  - run 출력 디렉터리 생성 helper
- `scripts/classification_report.py`
  - confusion matrix, per-category metric helper

## 먼저 읽을 파일

### prototype 전략 실험을 보고 싶을 때

1. `prototype_strategy_experiment.py`
2. `prototype_strategy/README.md`

### federated simulation을 보고 싶을 때

1. `run_federated_simulation.py`
2. `federated_simulation/README.md`

### seed / adaptation classifier를 보고 싶을 때

1. `train_softmax_classifier.py`
2. `train_lora_classifier.py`
3. `train_lora_pseudo_label_classifier.py`
4. `lora_classifier/runner.py`
5. 필요하면 `lora_classifier/query_adaptation_runner.py`, `lora_classifier/pseudo_label_runner.py`

## 주의할 점

- 이 디렉터리의 helper는 실험용 canonical shape를 만들 수는 있지만,
  `shared` 계약의 source of truth를 대체하지는 않는다.
- 실험에서 잘 된 로직을 운영 경로로 올릴 때는 해당 소유 경계
  (`shared/agent/main_server`)로 옮겨야 한다.
- config에 field가 있다고 곧바로 runtime 구현이 있다는 뜻은 아니다.
  secure aggregation/encryption 계열은 현재 typed contract는 있지만
  실제 runtime은 아직 붙지 않았다.
