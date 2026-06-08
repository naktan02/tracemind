# Agent Contracts

이 디렉터리는 agent-local API와 local collector가 주고받는 payload 계약을 소유한다.

소유 기준:

- raw text, page URL, 입력 surface hint처럼 agent local boundary 밖으로 나가면 안 되는
  값은 `agent/src/contracts/`에 둔다.
- `main_server`, `scripts`, 여러 runtime이 함께 해석해야 하는 FL/model/training 계약은
  `shared/src/contracts/`에 둔다.
- `apps/family_extension`은 이 계약을 소비해 타입을 생성하지만 계약 의미의 owner는 아니다.

현재 계약:

- `captured_text_contracts.py`: extension/local collector -> agent raw captured text
  ingest와 view generation debug job payload.
- `typing_segment_contracts.py`: browser/desktop typing segment -> agent ingest payload.
- `child_support_contracts.py`: family extension child-support UI -> agent support
  conversation/proactive prompt payload.
- `family_access_contracts.py`: family extension setup/unlock -> agent-local auth payload.
- `wellbeing_signal_contracts.py`: agent wellbeing projection -> family extension summary,
  timeseries, parent unlock payload.

`CapturedTextDebugJob*Payload`는 debug page가 agent-local view generation 상태를
조회하고 개발용 on/off, 즉시 실행을 요청하기 위한 payload다. production scheduler나
main_server orchestration 계약이 아니며, raw text와 generated weak/strong view는
agent local boundary에 남긴다.

Captured text 저장소는 raw event, view generation job/output, analysis job 상태를
분리한다. `captured_text_events`는 원문 snapshot만 소유하고, weak/strong view 생성
상태는 `captured_text_view_generation_jobs`, 후속 분류/분석 대기 상태는
`captured_text_analysis_jobs`가 소유한다.

Captured text ingest 응답의 `query_id`는 shape compatibility를 위해 `event_id`와 같은
값을 반환한다. ingest 경로는 분석 score를 생성하지 않으며, `top_category`와
`top_score`는 후처리 분석 job이 별도 저장소에 남길 때까지 `null`이다.

`ChildSupportConversation*` raw message와 conversation 저장도 agent-local 경계다.
LLM provider, prompt, safety policy, response plan 실행 방식은 agent runtime이
소유하고 UI는 응답을 표시한다.
