# Agent Training Runtime

`training_runtime/`는 server `TrainingTask`와 agent-local source를 selected
`methods/` core에 연결하는 runtime adapter 계층이다.

이 패키지는 SSL/FSSL 방법론 자체를 소유하지 않는다. 방법론의 local objective,
pseudo-label rule, PEFT update 계산 core는 `methods/`가 소유하고, agent는 다음
역할만 맡는다.

- current task 조회 결과를 agent-local 실행 요청으로 변환한다.
- captured text / analysis event 같은 local source를 training row로 정규화한다.
- active shared adapter state를 local trainer 입력으로 materialize한다.
- methods가 만든 update를 local artifact, envelope, server upload로 연결한다.

## 하위 패키지

- `current_task/`: live agent의 run-current-task application flow.
- `query_ssl/`: Query SSL current-task source selection, request build, upload,
  usage ledger orchestration.
- `query_ssl_peft/`: Query SSL/FSSL PEFT text encoder local update adapter.
- `storage/`: training artifact와 training source usage ledger 저장소.
- `training_sources/`: agent-local source를 Query SSL training row로 변환.

Legacy stored-event pseudo-label self-training path는 제거 대상이다. `top1_*`
selection hook은 논문용 Query SSL/FSSL method runtime의 기본 경로가 아니다.
