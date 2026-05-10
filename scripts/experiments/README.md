# Experiments

이 디렉터리는 코어 runtime을 조합해서 연구/검증을 수행하는 entrypoint layer다.

원칙:

- 운영 후보 algorithm/method core는 `methods`에 둔다.
- 공통 contract/domain은 `shared`, runtime adapter는 `agent`와 `main_server`에 둔다.
- 이 디렉터리는 그 코어를 조합하는 실험 CLI와 실험 전용 helper만 둔다.
- Hydra entrypoint config는 `conf/entrypoints/**/*.yaml`이 source of truth다.

전략 축 전체와 현재 override 가능 여부는
[docs/strategy_surface_map.md](../../docs/strategy_surface_map.md)를
먼저 보는 편이 빠르다.

## 구조

- `central_classifier_seed/`
  - 중앙 fixed embedding + classifier seed entrypoint를 둔다.
- `central_ssl_control/`
  - pooled/offline 중앙 SSL control entrypoint만 둔다.
  - LoRA SSL 구현체 자체를 소유하지 않는다.
- `fl_ssl/`
  - client split, round loop, aggregation, per-client metric 같은 FL SSL orchestration을 둔다.
- `prototype_analysis/`
  - prototype 전략 비교와 threshold sweep entrypoint/harness를 둔다.
- `fixed_classifier/`
  - fixed embedding classifier seed 학습과 artifact IO helper를 둔다.
- `query_lora_ssl/`
  - 중앙 SSL control과 FL client-local SSL에서 공유 가능한 LoRA 기반 query SSL harness를 둔다.

## 직접 실행하는 entrypoint

- `prototype_analysis/prototype_strategy_experiment.py`
  - single/kmeans/dbscan prototype 전략 비교.
- `prototype_analysis/prototype_threshold_sweep.py`
  - 선택된 prototype 전략 위에서 threshold policy 비교.
- `fl_ssl/run_federated_simulation.py`
  - agent/main_server 코어를 조합한 synthetic FL loop.
  - runtime/task/validation/report shape는
    `conf/entrypoints/fl_ssl/run_federated_simulation.yaml` 안의
    `round_runtime`, `training_task`, `validation`, `report` section이다.
  - `strategy_axes/fl/local_update_profile`은 local update backend,
    scoring/evidence, privacy 조합을 소유한다.
  - `strategy_axes/fl/round_runtime_profile`은 adapter family와 aggregation backend
    조합을 소유한다.
  - `strategy_axes/fl/method_descriptor`는 method identity/report metadata를
    소유하고, 실제 runtime 구현이 완료된 method만 열어야 한다.
- `central_classifier_seed/train_softmax_classifier.py`
  - 고정 임베딩 위 linear classifier baseline.
- `central_ssl_control/train_lora_classifier.py`
  - query-domain 적응 단계의 `frozen backbone + LoRA + classifier` canonical supervised baseline entrypoint.
- `central_ssl_control/train_lora_fixmatch.py`
  - USB `FixMatch` core를 같은 LoRA scaffold에 얹는 consistency baseline entrypoint.
  - method/source/augmentation/initial checkpoint source of truth는
    `strategy_axes/ssl/consistency_method`,
    `track_presets/central_ssl_control/query_source`,
    `strategy_axes/ssl/augmentation`,
    `strategy_axes/adaptation/initial_checkpoint` selector다.
- `central_ssl_control/train_lora_pseudolabel.py`
  - USB `PseudoLabel` core를 같은 LoRA scaffold에 얹는 weak-view SSL baseline entrypoint.
  - 원본 알고리즘과 같이 strong view를 요구하지 않으며 `text`를 weak view로 쓴다.
  - method/source/initial checkpoint source of truth는
    `strategy_axes/ssl/consistency_method`,
    `track_presets/central_ssl_control/query_source`,
    `strategy_axes/adaptation/initial_checkpoint` selector다.
- `central_ssl_control/train_lora_bootstrap_classifier_teacher.py`
  - 첫 pseudo-label 진입에서 `fixed embedding + classifier` teacher로 unlabeled pool에 pseudo-label을 붙이고,
    `LoRA + classifier` student를 학습하는 bootstrap entrypoint.
  - selection rule source of truth는 `conf/strategy_axes/ssl/pseudo_label_selection/*.yaml`이다.
  - student initial checkpoint source of truth는 `conf/strategy_axes/adaptation/initial_checkpoint/*.yaml`이다.
- `central_ssl_control/train_lora_pseudo_label_classifier.py`
  - 첫 bootstrap 이후 same-family `pseudo-label self-training`을 실행하는 entrypoint.
  - 현재 helper는 offline union retraining 경로를 포함하지만,
    central canonical 비교 규약은 `seed checkpoint 1회 생성 -> 이후 new accepted query-derived rows only continual adaptation`으로 본다.
  - 실험 표면에서는 bootstrap과 같은
    `strategy_axes/ssl/pseudo_label_selection` selector를 공유하고,
    산출물 manifest에 해당 preset을 provenance로 남긴다.
  - warm-start provenance는 `query_adaptation_initial_checkpoint` 축으로 함께 남긴다.
- `query_lora_ssl/`
  - query-domain LoRA scaffold의 helper 모듈
  - `runners/supervised.py`가 canonical supervised baseline runner다.
  - `query_ssl/common.py`가 Query SSL family 공통 scaffolding이다.
  - `runners/consistency.py`가 USB FixMatch를 포함한 consistency family runner다.
  - `query_ssl/augmentation.py`가 strict USB NLP `text + aug_0 + aug_1` preparation/cache를 담당한다.
  - `runners/query_adaptation.py`는 query adaptation dataset을 baseline runner에 연결하는 wrapper다.
  - `runners/pseudo_label.py`는 현재 offline union retraining helper를 제공한다.
  - `runners/bootstrap_teacher.py`의 teacher pseudo-label selection은
    `methods/ssl/hooks/`의 selection hook을 재사용한다.
  - bootstrap 실험에서 선택 규칙 preset은
    `strategy_axes/fl/local_update_profile`이 아니라
    `strategy_axes/ssl/pseudo_label_selection` selector로 고른다.
  - central canonical 비교 규약에서는 same initial checkpoint에서 출발해
    new accepted query-derived rows only continual adaptation으로 해석한다.
  - `io/query_adaptation.py`는 agent-local adaptation dataset을 현재 JSONL 입력 shape로 export한다.
  - `io/query_adaptation_multiview.py`는 필요 시 multiview dataset을 같은 JSONL shape로 export한다.
  - export는 `source_row.query_id`를 single source of truth로 쓰고, locale은 typed provenance에서 읽는다.
  - export는 JSONL/manifest와 함께 dataset summary JSON도 남긴다.
  - `runners/query_adaptation.py`는 labeled row를 메모리에서 바로 baseline runner에 넘기고, export 산출물은 trace/audit 용도로 함께 남긴다.

## 공통 helper

- `scripts/reporting/query_buffer_selection_diagnostics.py`
  - query-buffer selection summary/trace dump 저장 helper
- `scripts/artifacts/run_artifacts.py`
  - run 출력 디렉터리 생성 helper

## 먼저 읽을 파일

## `scripts/prototypes`와의 차이

- `scripts/experiments/prototype_analysis/*`
  - prototype 전략이나 threshold 정책을 비교하는 연구형 실험 레일
- `scripts/prototypes/*`
  - prototype pack을 실제로 seed/evaluate/pull/activate/report 하는
    artifact workflow 레일

즉 이름은 비슷하지만, 전자는 `비교/탐색`, 후자는 `artifact lifecycle`이 핵심이다.

### prototype 전략 실험을 보고 싶을 때

1. `prototype_analysis/prototype_strategy_experiment.py`
2. `prototype_analysis/prototype_strategy/README.md`

### federated simulation을 보고 싶을 때

1. `fl_ssl/run_federated_simulation.py`
2. `fl_ssl/federated_simulation/README.md`

### seed / adaptation classifier를 보고 싶을 때

1. `central_classifier_seed/train_softmax_classifier.py`
2. `central_ssl_control/train_lora_classifier.py`
3. `central_ssl_control/train_lora_pseudolabel.py`
4. `central_ssl_control/train_lora_fixmatch.py`
5. `central_ssl_control/train_lora_bootstrap_classifier_teacher.py`
6. `central_ssl_control/train_lora_pseudo_label_classifier.py`
7. `query_lora_ssl/runners/supervised.py`
8. 필요하면 `query_lora_ssl/runners/consistency.py`, `query_lora_ssl/runners/query_adaptation.py`, `query_lora_ssl/runners/bootstrap_teacher.py`, `query_lora_ssl/runners/pseudo_label.py`

## warm-start 재실행 요약

중앙 비교를 다시 돌릴 때는 아래 순서를 기준으로 본다.

1. `central_ssl_control/train_lora_classifier.py`
   - labeled seed subset으로 supervised LoRA seed manifest를 먼저 만든다.
2. `central_ssl_control/train_lora_bootstrap_classifier_teacher.py`
   - 기존 fixed classifier teacher로 pseudo-label을 붙이되,
     student는 `strategy_axes/adaptation/initial_checkpoint=required`와
     `query_adaptation_initial_checkpoint.manifest_path=<supervised_lora_manifest>`로
     같은 initial checkpoint에서 warm-start한다.
3. `central_ssl_control/train_lora_pseudolabel.py`
   - 같은 supervised LoRA seed manifest에서 warm-start하고,
     USB PseudoLabel처럼 unlabeled `text` weak view만 사용한다.
4. `central_ssl_control/train_lora_fixmatch.py`
   - 같은 supervised LoRA seed manifest에서 warm-start하고,
     기본값으로 materialized `labeled_train.with_views.jsonl` /
     `unlabeled_pool.with_views.jsonl`과
     `strategy_axes/ssl/augmentation=precomputed_usb_candidates_v1`를 사용한다.
     실행 중 역번역을 다시 만들지 않고 strict USB형 `text + aug_0 + aug_1`
     input이 없으면 실패한다.
   - `query_ssl_method.supervised_loss_weight=0.0`을 주면
     local labeled loss를 끄는 `unlabeled-only` ablation을 돌릴 수 있다.

현재 바로 복사해서 쓸 명령은 `scripts/README.md`의
중앙 LoRA/SSL 실행 명령과 각 entrypoint의 `--cfg job` preview를 기준으로 본다.

## 주의할 점

- 이 디렉터리의 helper는 실험용 canonical shape를 만들 수는 있지만,
  `shared` 계약의 source of truth를 대체하지는 않는다.
- 실험에서 잘 된 로직을 운영 경로로 올릴 때는 해당 소유 경계
  (`methods`, `shared`, `agent`, `main_server`, 필요 시 `conf`)로 옮겨야 한다.
- config에 field가 있다고 곧바로 runtime 구현이 있다는 뜻은 아니다.
  secure aggregation/encryption 계열은 현재 typed contract는 있지만
  실제 runtime은 아직 붙지 않았다.
