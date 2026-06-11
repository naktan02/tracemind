# Central SSL Control

중앙집중형 query-domain text encoder control entrypoint다. 이 트랙은
pooled/offline control이며, FL SSL non-IID 메인 비교를 대체하지 않는다.

긴 과거 cookbook은
`docs/notes/decisions/2026-05-28-archived-central-ssl-control-readme.md`에
보관했다. 현재 경계 판단은 이 README, `conf/README.md`,
`docs/contracts/central_peft_text_encoder_trainer_contract.md`를 기준으로 본다.

## 책임

- `run_peft_supervised_control.py`: PEFT text encoder scaffold의 supervised control
- `run_peft_ssl_control.py`: 같은 scaffold에서 SSL objective 비교
- `run_full_text_encoder_supervised_control.py`: full-model supervised-only ablation
- `build_method_projection_figure.py`: 완료된 report 기반 projection figure 생성
- `enrich_final_reports.py`: 과거 report의 final-selection metadata 보강

`scripts`는 dataset/report/artifact IO와 orchestration만 맡는다. SSL objective는
`methods/ssl`, PEFT local training core는
`methods/adaptation/peft_text_encoder`, full fine-tuning core는
`methods/adaptation/full_text_encoder`, 실행 조합은 루트 `conf/`가 소유한다.

## 읽기 경로

```text
conf/entrypoints/central/ssl_control/*.yaml
-> run_peft_supervised_control.py 또는 run_peft_ssl_control.py
-> scripts/support/query_ssl_text_encoder/runners/{supervised_text_encoder,consistency}.py
-> scripts/support/query_ssl_text_encoder/{text_encoder_run_context.py,query_ssl/run_context.py}
-> methods/adaptation/peft_text_encoder/training/{local_training_surface,query_ssl_training_session}.py
-> scripts/support/query_ssl_text_encoder/io/*
```

## 기본 실행

```bash
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py --cfg job
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py \
  run_controls/central_ssl/budget=smoke
uv run python scripts/experiments/central/ssl_control/run_peft_supervised_control.py \
  run_controls/central_ssl/budget=smoke
uv run python scripts/experiments/central/ssl_control/run_full_text_encoder_supervised_control.py \
  run_controls/central_ssl/budget=smoke
```

method는 `strategy_axes/ssl_objective/consistency_method`로 선택한다. PEFT
supervised/SSL entrypoint의 학습 표면은 `trainable_surface=peft_text_encoder`이고,
full-model supervised-only control은 `trainable_surface=full_text_encoder`를 쓴다.
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
실행 시에는 `query_data_selection.{labeled,unlabeled,validation,test}` selector만
override한다.

PEFT run은 `artifacts/adapter/`와 `artifacts/classifier_head.pt`, full text
encoder run은 `artifacts/model/`과 `artifacts/classifier_head.pt`를 저장한다.
`report.json`은 기존 downstream 호환을 위해 `results[test]`를 `best`로 유지하고,
마지막 epoch 기준 값은 `manifest.final_selection_report`와 `results.final`에 남긴다.

## 경계

이 폴더는 dataset, method, adapter family 기본값을 새로 정의하지 않는다. 중앙
Query SSL runner는 pooled/offline orchestration만 맡고, PEFT local SSL 학습은
FL/live와 같은 methods-owned local training surface를 통해 호출한다.
