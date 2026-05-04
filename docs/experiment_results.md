# TraceMind Experiment Results

이 문서는 현재 남아 있는 주요 실험 숫자를 짧게 보관한다. 구조와 계약의 source of
truth가 아니며, query-domain central LoRA/SSL 결과는 별도 run report가 생기면
추가한다.

## 주의

- 현재 숫자는 대부분 fixed embedding 또는 시스템 트랙 기준이다.
- 중앙 SSL control과 FL SSL main comparison은 같은 ranking으로 합치지 않는다.
- 세부 report는 각 `runs/<job>/<run_id>/` 아래 JSON을 source of truth로 본다.

## Canonical Baselines

| 실험 | artifact/report | 주요 설정 | 결과 | 해석 |
|---|---|---|---|---|
| PrototypePack single centroid | `data/processed/prototype_packs/proto_2026_03_28_163056.json`, `runs/prototype_pack_eval/<run_id>/reports/*` | `mxbai`, class당 prototype 1개 | validation accuracy `0.742794`, test accuracy `0.711694` | semantic bootstrap/comparison 기준점 |
| Fixed embedding + softmax classifier | `data/processed/classifier_heads/clf_2026_04_11_143138.pt`, `runs/train_classifier/clf_2026_04_11_143138/reports/report.json` | `mxbai`, linear head, CUDA, 12 epochs | validation accuracy `0.787744`, test accuracy `0.734879` | 현재 canonical seed artifact |

Canonical seed:

- id: `clf_2026_04_11_143138`
- model: `data/processed/classifier_heads/clf_2026_04_11_143138.pt`
- manifest: `data/processed/classifier_heads/clf_2026_04_11_143138.manifest.json`
- report: `runs/train_classifier/clf_2026_04_11_143138/reports/report.json`

## Prototype Strategy Comparison

Report:

- `runs/prototype_strategy/20260329T152634Z/summary.json`

| 전략 | validation accuracy | test accuracy | 메모 |
|---|---:|---:|---|
| `single` | `0.7427937915742794` | not selected | baseline |
| `kmeans` | `0.759322717194114` | `0.7096774193548387` | validation 기준 선택, class별 `k=2`, 총 8 prototypes |
| `dbscan` | `0.7397702076194316` | not selected | 총 19 prototypes, 정확도 이득 없음 |

해석:

- multi-prototype 분석에서 첫 선택지는 `kmeans`다.
- `dbscan`은 비교군 가치는 있지만 현재 기본 전략으로는 부적합하다.
- active runtime 기본값으로 multi-prototype을 올릴지는 별도 결정이 필요하다.

## Threshold Sweep

Purpose:

- pseudo-label 후보를 채택할 때 coverage와 purity를 비교한다.
- 주요 지표는 `accepted_ratio`, `accepted_accuracy`, `accepted_correct_ratio`다.

| 전략 | report | threshold | validation accepted accuracy | test accepted accuracy | 해석 |
|---|---|---|---:|---:|---|
| `kmeans` | `runs/prototype_threshold_sweep/20260329T155803Z/summary.json` | confidence `0.60`, margin `0.00` | `0.7461939973901697` | `0.7039337474120083` | coverage 우선, drift 위험 큼 |
| `single` | `runs/prototype_threshold_sweep/20260329T162700Z/summary.json` | confidence `0.60`, margin `0.00` | `0.7255075022065314` | `0.7023933402705516` | coverage-first에서도 kmeans보다 약함 |
| `kmeans` | `runs/prototype_threshold_sweep/20260329T163955Z/summary.json` | confidence `0.60`, margin `0.02` | `0.8808186195826645` | `0.8744588744588745` | 채택량 감소, purity 크게 증가 |
| `single` | `runs/prototype_threshold_sweep/20260329T164225Z/summary.json` | confidence `0.60`, margin `0.02` | `0.844097995545657` | `0.8579881656804734` | volume 쪽, purity는 kmeans보다 낮음 |

권장 해석:

- non-IID 초기 drift를 줄이려면 `kmeans + confidence 0.60 + margin 0.02` 쪽이 안전하다.
- volume을 우선하면 `single`이 더 많이 채택할 수 있지만 purity 손실이 있다.

## Smoke / 구조 검증

| 실험 | 대표 경로 | 확인한 것 |
|---|---|---|
| Prototype strategy smoke | `runs/prototype_strategy/<run_id>/summary.json` | hash_debug/real backend/refactor smoke 통과 |
| Threshold sweep smoke | `runs/prototype_threshold_sweep/<run_id>/summary.json` | grid search와 summary/grid 산출물 형식 |
| Federated simulation smoke | `runs/federated_simulation_smoke/20260331T155147Z/` | `train -> shard -> pseudo-label -> local update -> aggregation -> publication` 흐름 |

FL smoke에서 확인한 대표 산출물:

- `main_server/model_manifests/sim_rev_0000.json`
- `main_server/model_manifests/sim_rev_0001.json`
- `main_server/prototype_packs/proto_sim_0000.json`
- `main_server/prototype_packs/proto_sim_0001.json`
- `agents/agent_01/shared_adapter_updates/update_round_0001_08779ed1233e.json`

## 현재 권장안

1. 논문 seed 기준은 `clf_2026_04_11_143138`으로 고정한다.
2. 중앙 SSL은 pooled/offline control table로만 해석한다.
3. FL SSL non-IID main comparison은 별도 report track으로 저장한다.
4. prototype pseudo-label baseline은 `kmeans + margin` 조합을 우선 검토한다.
5. active runtime 기본값 변경은 FL SSL winner와 payload 설계가 확정된 뒤 판단한다.

## 다음 후보

- central SSL supervised/pseudo-label/prototype SSL/FixMatch/R-Drop/MixText control table.
- FL SSL non-IID smoke 재현.
- FedMatch/FedLGMatch/(FL)^2 후보 구현 범위 결정.
- LoRA/classifier-head family runtime translation 설계.
