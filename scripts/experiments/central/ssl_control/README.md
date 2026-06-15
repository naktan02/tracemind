# Central SSL Control

중앙집중형 query-domain text encoder control entrypoint다. 이 트랙은
pooled/offline control이며, FL SSL non-IID 메인 비교를 대체하지 않는다.

긴 과거 cookbook은
`docs/notes/decisions/2026-05-28-archived-central-ssl-control-readme.md`에
보관했다. 현재 경계 판단은 이 README, `conf/README.md`,
`docs/contracts/central_peft_text_encoder_trainer_contract.md`를 기준으로 본다.

## 책임

- `run_peft_supervised_control.py`: PEFT text encoder scaffold의 supervised control
- `run_query_ssl_control.py`: trainable surface를 바꿔 끼우는 SSL objective 비교
- `run_full_text_encoder_supervised_control.py`: full-model supervised-only ablation
- `build_method_projection_figure.py`: 완료된 report 기반 projection figure 생성
- `enrich_final_reports.py`: 과거 report의 final-selection metadata 보강

고정 feature 위 scikit-learn 지도학습 baseline은 별도 entrypoint인
`scripts/experiments/central/fixed_feature_control/`를 사용한다.

`scripts`는 dataset/report/artifact IO와 orchestration만 맡는다. SSL objective는
`methods/ssl`, text encoder/head 공통 학습 substrate는
`methods/adaptation/text_encoder_classifier`, PEFT surface와 FL update family는
`methods/adaptation/peft_text_encoder`, full fine-tuning surface는
`methods/adaptation/full_text_encoder`, 실행 조합은 루트 `conf/`가 소유한다.

## 읽기 경로

```text
conf/entrypoints/central/ssl_control/*.yaml
-> run_peft_supervised_control.py 또는 run_query_ssl_control.py
-> scripts/support/query_ssl_text_encoder/runners/{supervised_text_encoder,consistency}.py
-> scripts/support/query_ssl_text_encoder/{text_encoder_run_context.py,query_ssl/run_context.py}
-> methods/adaptation/text_encoder_classifier/{query_ssl_session.py,query_ssl_training.py}
-> methods/adaptation/{peft_text_encoder,full_text_encoder}/training/*_session.py
-> scripts/support/query_ssl_text_encoder/io/*
```

## 기본 실행

```bash
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py --cfg job
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  run_controls/central_ssl/budget=smoke
uv run python scripts/experiments/central/ssl_control/run_peft_supervised_control.py \
  run_controls/central_ssl/budget=smoke
uv run python scripts/experiments/central/ssl_control/run_full_text_encoder_supervised_control.py \
  run_controls/central_ssl/budget=smoke
uv run python scripts/experiments/central/ssl_control/run_peft_supervised_control.py \
  execution_context/query_labeled_budget=labeled100_per_class_seed42_nllb_views_v1 \
  central_ssl_budget.max_train_steps=2000 \
  output_dir=runs/central/supervised/peft_classifier_pc100_step2000
```

## 자주 바꾸는 실행 축

중앙 SSL 기본값은 `FixMatch + peft_text_encoder + main budget + pc1024`다.
`peft_text_encoder`는 LoRA/PEFT adapter와 classifier head만 학습하고,
`full_text_encoder`는 PEFT adapter 없이 encoder 전체와 classifier head를 학습한다.

```bash
# 기본 중앙 SSL: FixMatch + LoRA/PEFT + pc1024 + main budget
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py

# full text encoder로 중앙 SSL FixMatch 실행
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  strategy_axes/model_architecture/trainable_surface=full_text_encoder \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1 \
  run_controls/central_ssl/budget=main

# LoRA/PEFT surface를 명시해서 중앙 SSL FixMatch 실행
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  strategy_axes/model_architecture/trainable_surface=peft_text_encoder \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1 \
  run_controls/central_ssl/budget=main

# 라벨 예산만 pc1024 기본값에서 pc100 view artifact로 변경
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  execution_context/query_labeled_budget=labeled100_per_class_seed42_nllb_views_v1

# full text encoder + FixMatch + pc100
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  strategy_axes/model_architecture/trainable_surface=full_text_encoder \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1 \
  execution_context/query_labeled_budget=labeled100_per_class_seed42_nllb_views_v1 \
  run_controls/central_ssl/budget=main
```

method는 `strategy_axes/ssl_objective/consistency_method`로 선택한다.
`run_query_ssl_control.py`의 학습 표면은
`strategy_axes/model_architecture/trainable_surface`로 고른다. 기본값은
`peft_text_encoder`이고, LoRA/RS-LoRA/DoRA 같은 PEFT mechanism은
`strategy_axes/model_architecture/peft` 축이 고른다. `full_text_encoder`는 중앙
SSL/supervised control surface이며 FL update family는 아직 열지 않는다.
smoke 산출물은 main run과 섞지 않는다.

## Public Surface

사용자가 고르는 public surface는 아래로 제한한다.

- `strategy_axes/ssl_objective/consistency_method`
- `strategy_axes/model_architecture/{backbone,trainable_surface,peft,initial_checkpoint}`
- `execution_context/{dataset_asset,query_data_source,query_view,runtime_env}`
- `run_controls/central_ssl/budget`

`input_mode`, `teacher_provider`, `pseudo_label_selection`은 central SSL public
Hydra group이 아니다. 기본 중앙 supervised/SSL 실행은 `selection_set=test`이고
`eval_sets`에는 단일 `test`만 포함한다.

## 데이터와 산출물

source 주소록은 `conf/execution_context/query_data_source/default.yaml`이 소유한다.
실행 시에는 `query_data_selection.{labeled,unlabeled,validation,test}` selector를
override한다. 라벨 예산만 1024-per-class에서 FL main comparison의 shared pc100
labeled seed에서 추출한 pc100 view artifact로 바꿀 때는
`execution_context/query_labeled_budget=labeled100_per_class_seed42_nllb_views_v1`를
고른다.

PEFT run은 `artifacts/adapter/`와 `artifacts/classifier_head.safetensors`, full text
encoder run은 `artifacts/model/`과 `artifacts/classifier_head.safetensors`를 저장한다.
기존 `.pt` classifier head는 legacy warm-start reader에서만 읽는다.
`report.json`은 기존 downstream 호환을 위해 `results[test]`를 `best`로 유지하고,
마지막 epoch 기준 값은 `manifest.final_selection_report`와 `results.final`에 남긴다.

## 경계

이 폴더는 dataset, method, adapter family 기본값을 새로 정의하지 않는다. 중앙
Query SSL runner는 pooled/offline orchestration만 맡고, surface별 local SSL 학습과
artifact writer는 `trainable_surface.central_ssl.*` callable이 소유한다.
