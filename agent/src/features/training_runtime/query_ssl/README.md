# Query SSL Runtime

이 package는 live agent `TrainingTask`를 Query SSL/FSSL PEFT local update 실행으로
연결한다. SSL/FSSL 방법론 자체와 local objective 계산은 `methods/`가 소유하고, 이
package는 agent-local source, active shared adapter state, update upload, usage ledger
기록을 조립한다.

읽기 시작점:

- `task_service.py`: Query SSL current-task use case orchestration.
- `source_selection.py`: analysis event와 captured generated view를 training row로 선택.
- `method_request_builder.py`: selected method core가 읽는 request surface 조립.
- `upload_flow.py`: local update result를 server submission payload로 업로드.
- `usage_recording.py`: update에 사용된 source row ledger 기록.
