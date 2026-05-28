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
  - PEFT SSL 구현체 자체를 소유하지 않는다.
- `fl_ssl/`
  - client split, round loop, aggregation, per-client metric 같은 FL SSL orchestration을 둔다.
- `prototype_analysis/`
  - prototype 전략 비교와 threshold sweep entrypoint/harness를 둔다.
- `fixed_classifier/`
  - fixed embedding classifier seed 학습과 artifact IO helper를 둔다.
- `query_peft_ssl/`
  - 중앙 SSL control과 FL client-local SSL에서 공유 가능한 PEFT 기반 query SSL harness를 둔다.

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
  - `strategy_axes/trainable_state/update_family`가 update family와 v1 payload adapter
    compatibility alias를 선언하고, `round_runtime.aggregation_backend_name`은
    aggregation backend를 직접 고른다.
  - `strategy_axes/fl/method_descriptor`는 method identity/report metadata를
    소유하고, 실제 runtime 구현이 완료된 method만 열어야 한다.
- `central_classifier_seed/train_softmax_classifier.py`
  - 고정 임베딩 위 linear classifier baseline.
- `central_ssl_control/train_peft_supervised_classifier.py`
  - query-domain 적응 단계의 `frozen backbone + PEFT encoder classifier` canonical supervised baseline entrypoint.
- `central_ssl_control/train_peft_ssl_classifier.py`
  - USB `FixMatch`, `PseudoLabel` 등 Query SSL method를 같은 PEFT classifier scaffold에 얹는
    공통 SSL entrypoint.
  - `ssl_input_mode=consistency`는 `strategy_axes/ssl/consistency_method`를 실행한다.
  - `ssl_input_mode=pseudo_label_replay`는 이미 생성된 pseudo-label JSONL을
    같은 scaffold에서 replay/self-training 입력으로 사용한다.
  - method/source/augmentation/strong-view/initial checkpoint source of truth는
    `strategy_axes/ssl/consistency_method`,
    `execution_context/query_data_source`,
    `strategy_axes/ssl/augmentation_source`,
    `query_ssl_strong_view_policy`,
    `strategy_axes/adaptation/initial_checkpoint` selector다.
- `query_peft_ssl/`
  - query-domain PEFT classifier scaffold의 helper 모듈
  - `runners/supervised.py`가 canonical supervised baseline runner다.
  - `query_ssl/common.py`가 Query SSL family 공통 scaffolding이다.
  - `runners/consistency.py`가 USB FixMatch를 포함한 consistency family runner다.
  - `query_ssl/augmentation.py`가 strict USB NLP `text + aug_0 + aug_1` preparation/cache를 담당한다.
  - `runners/query_adaptation.py`는 query adaptation dataset을 baseline runner에 연결하는 wrapper다.
  - `runners/pseudo_label.py`는 pseudo-label replay/self-training helper를 제공한다.
  - `runners/bootstrap_teacher.py`의 teacher pseudo-label selection은
    `methods/ssl/hooks/`의 selection hook을 재사용한다.
  - bootstrap helper에서 선택 규칙 preset은
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
2. `central_ssl_control/train_peft_supervised_classifier.py`
3. `central_ssl_control/train_peft_ssl_classifier.py`
4. `query_peft_ssl/runners/supervised.py`
5. 필요하면 `query_peft_ssl/runners/consistency.py`, `query_peft_ssl/runners/query_adaptation.py`, `query_peft_ssl/runners/bootstrap_teacher.py`, `query_peft_ssl/runners/pseudo_label.py`

## warm-start 재실행 요약

중앙 비교를 다시 돌릴 때는 아래 순서를 기준으로 본다.

1. `central_ssl_control/train_peft_supervised_classifier.py`
   - labeled seed subset으로 supervised PEFT seed manifest를 먼저 만든다.
2. `central_ssl_control/train_peft_ssl_classifier.py`
   - 이미 생성된 pseudo-label JSONL을 쓰는 재생 실험은
     `ssl_input_mode=pseudo_label_replay`와 `pseudo_label_jsonl=<path>`로 같은
     entrypoint에서 실행한다.
3. `central_ssl_control/train_peft_ssl_classifier.py`
   - 같은 supervised PEFT seed manifest에서 warm-start하고,
     `strategy_axes/ssl/consistency_method=pseudolabel_usb_v1`로 USB PseudoLabel처럼
     unlabeled `text` weak view만 사용한다.
4. `central_ssl_control/train_peft_ssl_classifier.py`
   - 같은 supervised PEFT seed manifest에서 warm-start하고,
     기본 `strategy_axes/ssl/consistency_method=fixmatch_usb_v1`로 FixMatch를 실행한다.
     기본값으로 materialized `labeled_train.with_views.jsonl` /
     `unlabeled_pool.with_views.jsonl`과
     `strategy_axes/ssl/augmentation_source=precomputed_usb_candidates_v1`와
     `query_ssl_strong_view_policy=first_aug`를 사용한다.
     실행 중 역번역을 다시 만들지 않고 strict USB형 `text + aug_0 + aug_1`
     input이 없으면 실패한다.
   - `query_ssl_method.supervised_loss_weight=0.0`을 주면
     local labeled loss를 끄는 `unlabeled-only` ablation을 돌릴 수 있다.
   - FlexMatch, FreeMatch, AdaMatch는 같은 entrypoint에서
     `strategy_axes/ssl/consistency_method=<method>_usb_v1`만 바꿔 실행한다.
     AdaMatch는 labeled probability를 alignment/threshold에 쓰므로 labeled batch를
     끄는 ablation 대상은 아니다.

현재 바로 복사해서 쓸 명령은 `scripts/README.md`의
중앙 PEFT/SSL 실행 명령과 각 entrypoint의 `--cfg job` preview를 기준으로 본다.

## 주의할 점

- 이 디렉터리의 helper는 실험용 canonical shape를 만들 수는 있지만,
  `shared` 계약의 source of truth를 대체하지는 않는다.
- 실험에서 잘 된 로직을 운영 경로로 올릴 때는 해당 소유 경계
  (`methods`, `shared`, `agent`, `main_server`, 필요 시 `conf`)로 옮겨야 한다.
- config에 field가 있다고 곧바로 runtime 구현이 있다는 뜻은 아니다.
  secure aggregation/encryption 계열은 현재 typed contract는 있지만
  실제 runtime은 아직 붙지 않았다.
