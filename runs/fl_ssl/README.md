# FL SSL Runs

이 디렉터리는 FL SSL 실행 산출물을 method-first 구조로 보관한다.

- 생성일: 2026-05-18
- 기존 `runs/federated_simulation*` 산출물은 이 구조로 마이그레이션했고 원본
  수평 root는 제거했다.
- 산출물은 read-only evidence로 다루며, 새 실행도 같은 계층 아래에 쌓는다.

## Canonical Layout

```text
runs/fl_ssl/
  <method_family>/
    <method_or_composition>/
      <split_slug>/
        <clients_rounds_slug>/
          <run_id>/
        sweeps/
          <sweep_kind_rounds_slug>/
            <sweep_run_id>/
              <member_slug>/
```

현재 manual baseline은 아래 축을 쓴다.

```text
runs/fl_ssl/manual_baselines/
  fixmatch_usb_v1__lora_classifier__fedavg/
    alpha03_seed42/
      clients10_rounds50/
      clients10_rounds1/
      sweeps/client_count_rounds1/
```

## 역할

| 계층 | 의미 |
|---|---|
| `manual_baselines/` | lower-axis 조합으로 만든 manual FL baseline |
| `<method>__<adapter_family>__<aggregation>/` | 비교할 방법론/조합 |
| `alpha03_seed42/`, `alpha01_seed42/` | split/skew와 seed |
| `clients10_rounds50/` | client 수와 communication round budget |
| `sweeps/client_count_rounds1/` | 한 조합 안에서 client 수만 바꾼 sweep 묶음 |
| `incomplete/` | report가 없거나 중간에 멈춘 보존용 산출물 |
