# 2026-03-31 FL result interpretation

## User

```text
initial_model_revision=sim_rev_0000
initial_prototype_version=proto_sim_0000
initial_validation=accuracy:0.7404,accepted_ratio:0.5392
final_model_revision=sim_rev_0003
final_prototype_version=proto_sim_0003
final_validation=accuracy:0.7402,accepted_ratio:0.5410
round_count=3

fl 돌린결과인데 어떻게 해석해?
```

## Assistant

```text
현재 활성 문서 기준으로 이 결과는 shared representation 또는 shared adapter를
연합학습으로 조금 갱신한 뒤 validation이 어떻게 변했는지 보는 수치로 해석하는 게 맞다.

핵심 해석은 아래와 같다.

1. revision 진행
- model revision: sim_rev_0000 -> sim_rev_0003
- prototype version: proto_sim_0000 -> proto_sim_0003
- round_count=3 이므로 라운드가 돌 때마다 새 shared state와 prototype artifact가
  발행된 것으로 읽을 수 있다.

2. 성능 변화
- accuracy: 0.7404 -> 0.7402
- 절대 변화량: -0.0002
- 사실상 거의 동일한 수준이다.

- accepted_ratio: 0.5392 -> 0.5410
- 절대 변화량: +0.0018
- acceptance가 아주 미세하게 올라갔다.

3. 의미
- 3라운드 동안 모델이 크게 좋아졌다고 보기는 어렵다.
- 그렇다고 성능이 무너진 것도 아니다.
- 즉 지금 FL update는 shared layer를 약간 바꿨지만 validation 기준으로는
  거의 보합이다.

4. 조금 더 실무적으로 말하면
- accuracy는 그대로인데 accepted_ratio만 조금 올랐으므로,
  모델이 약간 더 permissive해졌을 가능성이 있다.
- 다만 변화폭이 너무 작아서 validation set 크기, seed variance,
  client sampling variance를 모르면 유의미한 차이라고 단정하면 안 된다.

5. 지금 단계에서의 좋은 신호
- revision이 3번 올라갔는데 accuracy drift가 거의 없다.
- 이는 aggregation이 불안정하게 폭주하진 않았다는 뜻으로 볼 수 있다.

6. 지금 단계에서의 아쉬운 신호
- "FL을 돌렸더니 성능이 개선됐다"라고 말할 근거는 없다.
- 현재 local objective, aggregation weight, learning rate, client heterogeneity 처리 중
  하나 이상이 아직 개선폭을 만들 만큼 강하지 않을 수 있다.

7. 다음으로 꼭 봐야 할 것
- round별 accuracy/accepted_ratio 추이
- client별 before/after 편차
- confusion matrix 또는 precision/recall 변화
- seed를 바꿔도 같은 경향이 나오는지

한 줄 요약:
이번 결과는 "3라운드 FL 후 shared model/prototype revision은 갱신됐지만,
validation 성능은 사실상 보합이고 acceptance만 아주 미세하게 증가했다"로
해석하는 게 가장 안전하다.
```

## User

```text
진행해봐
```

## Assistant

```text
같은 실험의 round별 흔적과 metric 산출 경로를 먼저 찾겠다고 안내했다.
`run_federated_simulation.py`, 설정 YAML, `tmp/federated_simulation` 산출물,
validation 크기, 누적된 update 파일 패턴을 확인했다.

이후 4,961건 validation 전체를 재평가하려고 `.venv/bin/python`으로
재계산을 시작했는데, 로컬 런타임이 `cpu`로 잡혀 CPU 배치 인퍼런스가 진행되었다.
```

## User

```text
난 cpu로 안 돌릴거야 cpu 실행중인거 종료시켜
```

## Assistant

```text
내가 띄운 `.venv/bin/python` PTY 세션에 `Ctrl-C`를 보내 CPU 실행을 중단했다.
이후 `ps -p 70213,70214 -o pid=,stat=,comm=`로 확인했을 때 남아 있는 프로세스가
없어서 종료된 상태임을 확인했다.
```

## User

```text
내가 실행했었던거 알려줄게 이걸 토대로 tmp에서 기록 찾아서 들어가서 확인해봐

(tracemind-server) jmgjmg102@DESKTOP-VVOL8MH:~/tracemind_server$   uv run python scripts/experiments/run_federated_simulation.py \
    federated_run_preset=standard \
    federated_run_preset.rounds=3 \
    federated_run_preset.client_count=4
...
initial_model_revision=sim_rev_0000
initial_prototype_version=proto_sim_0000
initial_validation=accuracy:0.7404,accepted_ratio:0.5392
final_model_revision=sim_rev_0003
final_prototype_version=proto_sim_0003
final_validation=accuracy:0.7402,accepted_ratio:0.5410
round_count=3
```

## Assistant

```text
재실행 없이 `tmp/federated_simulation` 산출물만 읽어서 2026-03-31 02:30~02:35 KST에
해당하는 파일 구간을 분리했다. 같은 디렉터리에 02:17, 02:26 실행 흔적도 함께
남아 있어서, 이번 run은 02:30 이후 파일만 기준으로 봐야 한다.

확인한 핵심은 아래와 같다.

1. 이번 run의 main_server 산출물 타임라인
- 02:30:15 `sim_rev_0000` state 저장
- 02:30:29 `sim_rev_0000` manifest와 `proto_sim_0000` 저장
- 02:31:52 `sim_rev_0001` / `proto_sim_0001`
- 02:33:27 `sim_rev_0002` / `proto_sim_0002`
- 02:35:04 `sim_rev_0003`
- 02:35:18 `proto_sim_0003`

2. batch 로그와 실제 shard 크기가 정확히 대응한다
- bootstrap rows: 8,930 -> 559 batches
- validation rows: 4,961 -> 311 batches
- agent_01 shard: 4,990 -> 312 batches
- agent_02 shard: 8,824 -> 552 batches
- agent_03 shard: 12,542 -> 784 batches
- agent_04 shard: 9,365 -> 586 batches

3. 모든 라운드에서 4개 client가 모두 update를 냈고, 각 client는 매번 64개씩만 반영됐다
- 이는 `standard` preset의 `max_examples=64` cap과 맞는다.
- round당 총 반영 예시는 256개다.

4. 더 중요한 점은 update 조성이 3라운드 내내 사실상 고정돼 있었다
- round_0001 totals: anxiety 14, depression 71, suicidal 171
- round_0002 totals: anxiety 14, depression 71, suicidal 171
- round_0003 totals: anxiety 14, depression 71, suicidal 171
- `normal` pseudo-label은 3라운드 전체에서 한 번도 선택되지 않았다.

5. shard 분포와 비교하면 selection 편향이 보인다
- agent_03 shard는 depression 2,611 / normal 9,931인데 update는 depression 37 / suicidal 23 / anxiety 4
- agent_04 shard는 normal 3,311 / suicidal 6,054인데 update는 매 라운드 suicidal 64 only
- 즉 현재 threshold + heuristic update 조합은 normal을 거의 학습 신호로 쓰지 못하고 있다.

6. shared adapter drift는 누적되지만 작다
- sim_rev_0001: mean_abs_from_rev0 0.001061, max_abs 0.009688
- sim_rev_0002: mean_abs_from_rev0 0.002122, max_abs 0.019467
- sim_rev_0003: mean_abs_from_rev0 0.003183, max_abs 0.029337
- 1024차원 중 below 1.0 = 518, above 1.0 = 506 으로 부호 패턴도 사실상 유지된다.

7. prototype도 매 라운드 재생성되지만 이동량은 매우 작다
- sample_count는 전 버전 동일: anxiety 991, depression 2611, normal 3310, suicidal 2018
- `proto_sim_0000 -> proto_sim_0003` centroid cosine
  - anxiety 0.99995150
  - depression 0.99995184
  - normal 0.99994315
  - suicidal 0.99995349
- 즉 prototype version은 올라가지만 위치 변화는 미세하다.

8. 따라서 이번 flat metric은 tmp 기록만 봐도 설명된다
- 각 round의 client update가 거의 같은 분포와 비슷한 confidence/margin으로 반복된다.
- adapter와 prototype drift도 작다.
- 그래서 3라운드가 서로 다른 학습 단계라기보다, 거의 같은 update를 세 번 누적 적용한 것에 가깝다.
```

## User

```text
normal을 학습신호로 왜 못 쓰는데?
```

## Assistant

```text
코드상으로 `normal`을 금지하는 로직은 없다. 문제는 현재 pseudo-label selection 정책이
`normal`을 거의 최종 채택에 못 올린다는 데 있다.

근거:

1. [`agent/src/services/training/pseudo_label_service.py`](/home/jmgjmg102/tracemind_server/agent/src/services/training/pseudo_label_service.py)
   - top1 label만 pseudo-label 후보가 된다.
   - `confidence >= 0.6` and `margin >= 0.02`를 만족해야 accepted 된다.
   - 그 뒤 accepted 후보를 confidence, margin 순으로 정렬해서 `max_examples`까지만 남긴다.

2. [`scripts/conf/experiments/run_federated_simulation.yaml`](/home/jmgjmg102/tracemind_server/scripts/conf/experiments/run_federated_simulation.yaml)
   - 이번 run threshold는 `confidence_threshold=0.6`, `margin_threshold=0.02`

3. [`scripts/conf/federated_run_preset/standard.yaml`](/home/jmgjmg102/tracemind_server/scripts/conf/federated_run_preset/standard.yaml)
   - client당 최대 반영 예시는 `64`

4. 따라서 `normal`이 빠질 수 있는 지점은 두 군데다.
   - 아예 threshold를 못 넘는다.
   - threshold는 넘겨도 confidence/margin 순위에서 top 64 밖으로 밀린다.

5. 이번 run 산출물은 실제로 두 번째 현상까지 강하게 시사한다.
   - agent_03 shard는 `normal 9,931`이었는데 round_0001 update는 `depression 37`, `suicidal 23`, `anxiety 4`, `normal 0`
   - agent_04 shard는 `normal 3,311 / suicidal 6,054`인데 round_0001 update는 `suicidal 64` only
   - 즉 `normal`이 많아도, 현재 점수 체계에서 더 높은 confidence/margin을 받은 건 mostly depression/suicidal이었다.

6. 중요한 보정:
   - 지금 기록만으로는 `normal`이 threshold 자체를 거의 못 넘었는지,
     아니면 넘긴 뒤에도 top 64에서 탈락했는지까지는 확정할 수 없다.
   - 그걸 알려면 per-row score/confidence/margin dump가 필요하다.

따라서 정확한 표현은
`normal을 못 쓴다`가 아니라
`현재 prototype scoring + threshold + top64 cap 조합에서 normal이 거의 채택되지 않는다`가 맞다.
```
