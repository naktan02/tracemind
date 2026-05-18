# FL SSL Legacy Runs

이 디렉터리는 기존 `runs/federated_simulation*` 산출물을 신규 FL SSL 계층 아래로
마이그레이션한 read-only legacy archive다.

- 생성일: 2026-05-18
- 방식: `cp -al`로 새 계층을 만든 뒤 기존 수평 FL root 제거
- 원본 삭제: 완료
- 주의: legacy evidence이므로 artifact 내용은 수정하지 않는다.

## 분류

| 새 위치 | 원본 의미 |
|---|---|
| `evidence/main/` | alpha=0.3, 10 clients, 50 rounds 완료 evidence |
| `evidence/reduced/` | 5-round reduced 비교 evidence |
| `evidence/short/` | 1-round quick 비교 evidence |
| `sweeps/client_count/short/` | 1-round client-count sweep 완료 evidence |
| `sweeps/client_count/reduced_incomplete/` | 5-round client-count sweep 부분 실행 |
| `smoke/preflight/` | main split preflight |
| `smoke/runtime/` | runtime metadata / artifact smoke |
| `smoke/materialized_cpu/` | materialized split 초기 CPU/debug smoke |
| `smoke/materialized_gpu/` | materialized split 초기 GPU-named smoke |
| `incomplete/` | report가 없는 incomplete long run |

## 원본 Root

- `runs/federated_simulation`
- `runs/federated_simulation_main_preflight`
- `runs/federated_simulation_reduced`
- `runs/federated_simulation_short`
- `runs/federated_simulation_smoke`
- `runs/federated_simulation_materialized_smoke`
- `runs/federated_simulation_materialized_gpu_smoke`
- `runs/federated_simulation_client_count_sweep_short`
- `runs/federated_simulation_client_count_sweep_reduced`
