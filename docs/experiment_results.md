# TraceMind Experiment Results

이 문서는 현재 남아 있는 대표 결과의 짧은 지도다. 구조, 계약, 실행 기본값의 source
of truth가 아니며 세부 숫자는 각 `runs/<track>/.../reports/*` artifact가 소유한다.
긴 과거 표는 `docs/notes/decisions/2026-05-28-archived-experiment-results.md`에
보관했다.

## 해석 규칙

- 중앙 SSL control과 FL SSL main comparison은 같은 ranking table로 합치지 않는다.
- fixed embedding/prototype 결과는 bootstrap 기준점으로만 해석한다.
- FL SSL 결과는 non-IID client split, seed, budget, method descriptor가 명시된 report만
  비교한다.
- 현재 로컬에 없는 historical artifact는 active current-result 표에 올리지 않는다.

## Canonical Seed

- id: `clf_2026_04_11_143138`
- model: `data/processed/classifier_heads/clf_2026_04_11_143138.pt`
- manifest: `data/processed/classifier_heads/clf_2026_04_11_143138.manifest.json`
- report: `runs/train_classifier/clf_2026_04_11_143138/reports/report.json`
- validation accuracy: `0.787744`
- test accuracy: `0.734879`

## Prototype Baselines

- single centroid: validation accuracy `0.742794`, test accuracy `0.711694`.
- kmeans multi-prototype: validation accuracy `0.759323`, test accuracy `0.709677`.
- pseudo-label threshold 후보는 `kmeans + confidence 0.60 + margin 0.02`를 우선 검토한다.

## FL SSL Current Result Gate

- 감사표: `docs/notes/sessions/2026-05-28-archived-fl-ssl-execution-audit.md`
- 검증 manifest: `docs/operations/fl_ssl_artifact_verification_manifest.current.json`
- 검증 명령:

```bash
uv run python scripts/experiments/fl_ssl/verify_federated_report_artifacts.py \
  --manifest docs/operations/fl_ssl_artifact_verification_manifest.current.json
```

## 다음 기준

1. 논문 seed 기준은 `clf_2026_04_11_143138`로 고정한다.
2. 중앙 SSL은 pooled/offline control table로만 해석한다.
3. FL SSL non-IID main comparison은 별도 report track으로 저장한다.
4. 새 결과를 추가할 때는 report path, split policy, seed, budget, metric을 함께 적는다.
