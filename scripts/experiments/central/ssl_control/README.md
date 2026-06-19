# Central SSL Experiments

중앙집중형 query-domain text encoder 실험 entrypoint다. 이 트랙은 모든 데이터를 한
곳에 모아 실행하는 pooled/offline control이며, FL SSL non-IID 메인 비교를 대체하지
않는다.

## When To Use

- PEFT/LoRA text encoder 지도학습 baseline을 만들 때
- full text encoder 지도학습 ablation을 만들 때
- FixMatch, AdaMatch, FreeMatch 같은 SSL objective를 같은 중앙 조건에서 비교할 때
- FL SSL 실험 전에 supervised checkpoint나 중앙 control table이 필요할 때

고정 feature 위 scikit-learn 지도학습 baseline은
`scripts/experiments/central/fixed_feature_control/README.md`를 사용한다.

## Entrypoints

| Goal | Entrypoint |
|---|---|
| PEFT/LoRA text encoder 지도학습 | `run_peft_supervised_control.py` |
| full text encoder 지도학습 | `run_full_text_encoder_supervised_control.py` |
| 중앙 SSL objective 비교 | `run_query_ssl_control.py` |
| 완료된 report 기반 projection figure 생성 | `build_method_projection_figure.py` |
| 과거 report final-selection metadata 보강 | `enrich_final_reports.py` |

실행 전 Hydra compose 결과를 먼저 확인할 수 있다.

```bash
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py --cfg job
```

## Standard Run Shape

기본 중앙 SSL 실행은 아래 조합이다.

```text
SSL objective: FixMatch USB v1
trainable surface: peft_text_encoder
backbone: mxbai encoder
labeled budget: pc1024
budget: main
selection/eval set: test
```

`peft_text_encoder`는 LoRA/PEFT adapter와 classifier head만 학습한다.
`full_text_encoder`는 PEFT adapter 없이 encoder 전체와 classifier head를 학습한다.

## Choose A Run

| If you want to... | Use |
|---|---|
| 가장 빠르게 wiring을 확인한다 | `run_controls/central_ssl/budget=smoke` |
| 기본 중앙 SSL control을 실행한다 | `run_query_ssl_control.py` |
| 지도학습 PEFT checkpoint를 만든다 | `run_peft_supervised_control.py` |
| full fine-tuning ablation을 본다 | `run_full_text_encoder_supervised_control.py` |
| 라벨 예산만 줄인다 | `execution_context/query_labeled_budget=labeled100_per_class_seed42_nllb_views_v1` |
| SSL method만 바꾼다 | `strategy_axes/ssl_objective/consistency_method=<method>` |
| 학습 surface를 바꾼다 | `strategy_axes/model_architecture/trainable_surface=<surface>` |

## Combination Axes

| Axis | Common values |
|---|---|
| SSL objective | `fixmatch_usb_v1`, `adamatch_usb_v1`, `freematch_usb_v1`, `pseudolabel_usb_v1` |
| Trainable surface | `peft_text_encoder`, `full_text_encoder` |
| Backbone | `mxbai_encoder`, `roberta_base_v2` |
| PEFT config | `default` |
| Initial checkpoint | `none`, `supervised_20260612_step2000` |
| Labeled budget | `per_class1024`, `labeled100_per_class_seed42_nllb_views_v1` |
| Runtime | `gpu_local`, `gpu_online`, `cpu_local` |
| Budget | `smoke`, `main` |

Public override surface는 아래 범위로 제한한다.

- `strategy_axes/ssl_objective/consistency_method`
- `strategy_axes/model_architecture/{backbone,trainable_surface,peft,initial_checkpoint}`
- `execution_context/{dataset_asset,query_data_source,query_view,runtime_env}`
- `execution_context/query_labeled_budget`
- `run_controls/central_ssl/budget`

`input_mode`, `teacher_provider`, `pseudo_label_selection`은 central SSL public Hydra
group이 아니다.

## Quick Examples

### Supervised Controls

```bash
# PEFT/LoRA text encoder + classifier head 지도학습 smoke.
uv run python scripts/experiments/central/ssl_control/run_peft_supervised_control.py \
  run_controls/central_ssl/budget=smoke

# PEFT/LoRA text encoder + classifier head 지도학습 main.
uv run python scripts/experiments/central/ssl_control/run_peft_supervised_control.py \
  run_controls/central_ssl/budget=main

# full text encoder + classifier head 지도학습 smoke.
uv run python scripts/experiments/central/ssl_control/run_full_text_encoder_supervised_control.py \
  run_controls/central_ssl/budget=smoke

# pc100 labeled view로 PEFT supervised reduced 비교.
uv run python scripts/experiments/central/ssl_control/run_peft_supervised_control.py \
  execution_context/query_labeled_budget=labeled100_per_class_seed42_nllb_views_v1 \
  central_ssl_budget.max_train_steps=2000 \
  output_dir=runs/central/supervised/peft_classifier_pc100_step2000
```

### SSL Controls

```bash
# 기본 중앙 SSL: FixMatch + LoRA/PEFT + pc1024 + main budget.
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py

# FixMatch + LoRA/PEFT + smoke.
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1 \
  run_controls/central_ssl/budget=smoke

# FixMatch + full text encoder + main budget.
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  strategy_axes/model_architecture/trainable_surface=full_text_encoder \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1 \
  run_controls/central_ssl/budget=main

# FixMatch + LoRA/PEFT + pc100 labeled view.
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1 \
  execution_context/query_labeled_budget=labeled100_per_class_seed42_nllb_views_v1

# AdaMatch + LoRA/PEFT + smoke.
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  strategy_axes/ssl_objective/consistency_method=adamatch_usb_v1 \
  run_controls/central_ssl/budget=smoke

# FreeMatch + RoBERTa backbone + LoRA/PEFT + smoke.
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  strategy_axes/model_architecture/backbone=roberta_base_v2 \
  strategy_axes/ssl_objective/consistency_method=freematch_usb_v1 \
  run_controls/central_ssl/budget=smoke
```

## Outputs

PEFT run은 아래 artifact를 저장한다.

```text
artifacts/adapter/
artifacts/classifier_head.safetensors
reports/report.json
projections/
```

Full text encoder run은 아래 artifact를 저장한다.

```text
artifacts/model/
artifacts/classifier_head.safetensors
reports/report.json
projections/
```

`report.json`은 downstream 호환을 위해 `results[test]`를 `best`로 유지한다. 마지막
epoch 기준 값은 `manifest.final_selection_report`와 `results.final`에 남긴다.

## Internal References

사람이 코드를 읽을 때는 아래 순서가 가장 짧다.

```text
conf/entrypoints/central/ssl_control/*.yaml
-> run_peft_supervised_control.py 또는 run_query_ssl_control.py
-> scripts/support/query_ssl_text_encoder/runners/{supervised_text_encoder,consistency}.py
-> scripts/support/query_ssl_text_encoder/{text_encoder_run_context.py,query_ssl/run_context.py}
-> methods/adaptation/text_encoder_classifier/{query_ssl_session.py,query_ssl_training.py}
-> methods/adaptation/{peft_text_encoder,full_text_encoder}/training/*_session.py
-> scripts/support/query_ssl_text_encoder/io/*
```

긴 과거 cookbook은
`docs/notes/decisions/2026-05-28-archived-central-ssl-control-readme.md`에 보관했다.
현재 경계 판단은 이 README, `conf/README.md`,
`docs/contracts/central_peft_text_encoder_trainer_contract.md`를 기준으로 본다.

이 폴더는 dataset, method, adapter family 기본값을 새로 정의하지 않는다. 중앙
Query SSL runner는 pooled/offline orchestration만 맡고, surface별 local SSL 학습과
artifact writer는 `trainable_surface.central_ssl.*` callable이 소유한다.
