# 2026-03-31 FL result interpretation 요약

이 파일은 archive-only 세션 요약이다. 현재 FL 실행 기준은
`docs/project_execution_plan.md`와 `docs/fl_runtime_implementation_checklist.md`를 본다.

## 해석한 결과

사용자가 공유한 3-round FL simulation 결과:

```text
initial_model_revision=sim_rev_0000
final_model_revision=sim_rev_0003
initial_prototype_version=proto_sim_0000
final_prototype_version=proto_sim_0003
initial_validation=accuracy:0.7404, accepted_ratio:0.5392
final_validation=accuracy:0.7402, accepted_ratio:0.5410
round_count=3
```

안전한 해석:

- revision과 prototype artifact는 3라운드 동안 정상적으로 갱신됐다.
- validation accuracy 변화는 `-0.0002`로 사실상 보합이다.
- accepted ratio 변화는 `+0.0018`로 미세 증가다.
- 개선 근거로 말하기에는 약하고, aggregation이 폭주하지 않았다는 smoke 신호로 보는 편이 맞다.

## 당시 확인한 점

- 재실행 없이 기존 산출물을 읽어 run timeline과 revision 발행을 확인했다.
- CPU 재평가가 의도와 달라 중단했고, GPU preflight 후 실행해야 한다는 운영 규칙을 남겼다.
- 이후 실행 결과 해석은 round별 metric, client별 편차, confusion/precision/recall, seed variance를 같이 봐야 한다고 정리했다.

## 현재 반영 위치

- FL simulation harness: `scripts/experiments/federated_simulation/*`
- FL runtime checklist: `docs/fl_runtime_implementation_checklist.md`
- local runbook/GPU preflight: `docs/operations/local-runbook.md`
