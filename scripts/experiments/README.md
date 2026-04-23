# Experiments

이 디렉터리는 코어 runtime을 조합해서 연구/검증을 수행하는 entrypoint layer다.

원칙:

- 운영 후보 로직은 `shared`, `agent`, `main_server`에 둔다.
- 이 디렉터리는 그 코어를 조합하는 실험 CLI와 실험 전용 helper만 둔다.
- Hydra top-level config는 `scripts/conf/experiments/*.yaml`이 source of truth다.

전략 축 전체와 현재 override 가능 여부는
[docs/strategy_surface_map.md](....docs/strategy_surface_map.md)를
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
- `train_lora_fixmatch.py`
  - USB `FixMatch` core를 같은 LoRA scaffold에 얹는 consistency baseline entrypoint
  - method/source/augmentation/initial checkpoint source of truth는 `query_ssl_method`, `query_ssl_train_source`, `query_ssl_augmenter`, `query_adaptation_initial_checkpoint` Hydra group이다.
- `train_lora_bootstrap_classifier_teacher.py`
  - 첫 pseudo-label 진입에서 `fixed embedding + classifier` teacher로 unlabeled pool에 pseudo-label을 붙이고,
    `LoRA + classifier` student를 학습하는 bootstrap entrypoint
  - selection rule source of truth는 `scripts/conf/pseudo_label_algorithm/*.yaml`이다.
  - student initial checkpoint source of truth는 `scripts/conf/query_adaptation_initial_checkpoint/*.yaml`이다.
- `train_lora_pseudo_label_classifier.py`
  - 첫 bootstrap 이후 same-family `pseudo-label self-training`을 실행하는 entrypoint
  - 현재 helper는 offline union retraining 경로를 포함하지만,
    central canonical 비교 규약은 `seed checkpoint 1회 생성 -> 이후 new accepted query-derived rows only continual adaptation`으로 본다.
  - 실험 표면에서는 bootstrap과 같은 `pseudo_label_algorithm` Hydra group을 공유하고,
    산출물 manifest에 해당 preset을 provenance로 남긴다.
  - warm-start provenance는 `query_adaptation_initial_checkpoint` 축으로 함께 남긴다.
- `lora_classifier/`
  - query-domain LoRA scaffold의 helper 모듈
  - `runner.py`가 canonical supervised baseline runner다.
  - `query_ssl/common.py`가 Query SSL family 공통 scaffolding이다.
  - `query_ssl/consistency_runner.py`가 USB FixMatch를 포함한 consistency family runner다.
  - `query_ssl/augmentation.py`가 strict USB NLP `text + aug_0 + aug_1` preparation/cache를 담당한다.
  - `query_adaptation_runner.py`는 query adaptation dataset을 baseline runner에 연결하는 wrapper다.
  - `pseudo_label_runner.py`는 현재 offline union retraining helper를 제공한다.
  - `bootstrap_runner.py`의 teacher pseudo-label selection은
    `agent/src/services/training/query_adaptation/ssl/algorithms/` 코어를 재사용한다.
  - bootstrap 실험에서 선택 규칙 preset은
    `training_algorithm_profile`이 아니라 `pseudo_label_algorithm` Hydra group으로 고른다.
  - central canonical 비교 규약에서는 same initial checkpoint에서 출발해
    new accepted query-derived rows only continual adaptation으로 해석한다.
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

## `scripts/prototypes`와의 차이

- `scripts/experiments/prototype_*`
  - prototype 전략이나 threshold 정책을 비교하는 연구형 실험 레일
- `scripts/prototypes/*`
  - prototype pack을 실제로 seed/evaluate/pull/activate/report 하는
    artifact workflow 레일

즉 이름은 비슷하지만, 전자는 `비교/탐색`, 후자는 `artifact lifecycle`이 핵심이다.

### prototype 전략 실험을 보고 싶을 때

1. `prototype_strategy_experiment.py`
2. `prototype_strategy/README.md`

### federated simulation을 보고 싶을 때

1. `run_federated_simulation.py`
2. `federated_simulation/README.md`

### seed / adaptation classifier를 보고 싶을 때

1. `train_softmax_classifier.py`
2. `train_lora_classifier.py`
3. `train_lora_fixmatch.py`
4. `train_lora_bootstrap_classifier_teacher.py`
5. `train_lora_pseudo_label_classifier.py`
6. `lora_classifier/runner.py`
7. 필요하면 `lora_classifier/query_ssl/consistency_runner.py`, `lora_classifier/query_adaptation_runner.py`, `lora_classifier/bootstrap_runner.py`, `lora_classifier/pseudo_label_runner.py`

## warm-start 재실행 요약

중앙 비교를 다시 돌릴 때는 아래 순서를 기준으로 본다.

1. `train_lora_classifier.py`
   - labeled seed subset으로 supervised LoRA seed manifest를 먼저 만든다.
2. `train_lora_bootstrap_classifier_teacher.py`
   - 기존 fixed classifier teacher로 pseudo-label을 붙이되,
     student는 `query_adaptation_initial_checkpoint=required`와
     `query_adaptation_initial_checkpoint.manifest_path=<supervised_lora_manifest>`로
     같은 initial checkpoint에서 warm-start한다.
3. `train_lora_fixmatch.py`
   - 같은 supervised LoRA seed manifest에서 warm-start하고,
     `query_ssl_augmenter=backtranslation_nllb_en_de_fr_usb_v1`로
     strict USB형 `text + aug_0 + aug_1` input을 생성한다.
   - `query_ssl_method.supervised_loss_weight=0.0`을 주면
     local labeled loss를 끄는 `unlabeled-only` ablation을 돌릴 수 있다.

현재 바로 복사해서 쓸 명령은 `scripts/README.md`의
`현재 warm-start 재실행 레시피` 절에 정리해 둔다.

## 주의할 점

- 이 디렉터리의 helper는 실험용 canonical shape를 만들 수는 있지만,
  `shared` 계약의 source of truth를 대체하지는 않는다.
- 실험에서 잘 된 로직을 운영 경로로 올릴 때는 해당 소유 경계
  (`shared/agent/main_server`)로 옮겨야 한다.
- config에 field가 있다고 곧바로 runtime 구현이 있다는 뜻은 아니다.
  secure aggregation/encryption 계열은 현재 typed contract는 있지만
  실제 runtime은 아직 붙지 않았다.
