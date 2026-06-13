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
- `wellbeing_space_web_contracts.py`: agent wellbeing category projection -> family
  extension child analysis space-web nodes/edges payload.

`CapturedTextDebugJob*Payload`는 debug page가 agent-local view generation/analysis
상태를 조회하고 개발용 on/off, 즉시 실행을 요청하기 위한 payload다. production
scheduler나 main_server orchestration 계약이 아니며, raw text와 generated weak/strong
view는 agent local boundary에 남긴다.

Captured text, analysis event, training usage ledger의 기본 저장소는 같은 agent-local
SQLite 파일(`agent_local.db`)을 사용한다. `captured_text_events`는 원문 snapshot만
소유하고, weak/strong view 생성 상태는 `captured_text_view_generation_jobs`,
generated weak/strong view는 `captured_text_generated_views`, 후속 분류/분석 상태는
`captured_text_analysis_jobs`가 소유한다. completed analysis job은
`analysis_events.analysis_id`를 참조한다. 학습 실행 이력은
`training_usage_runs`와 `training_usage_rows`가 round/task/update와 사용된 source row를
연결한다.

Captured text ingest 응답의 `query_id`는 shape compatibility를 위해 `event_id`와 같은
값을 반환한다. ingest 경로는 분석 score를 생성하지 않으며, `top_category`와
`top_score`는 debug/후처리 분석 job이 `analysis_events`에 score를 남길 때까지
`null`이다.

`ChildSupportConversation*` raw message와 conversation 저장도 agent-local 경계다.
LLM provider, prompt, safety policy, response plan 실행 방식은 agent runtime이
소유하고 UI는 응답을 표시한다. 로컬 LLM 호출은 provider adapter 뒤에서
구조화된 `system`/`user`/`assistant` message로 전달하며, 단일 prompt 문자열은
debug와 provider fallback 표면으로만 유지한다.
`ChildSupportProactivePromptPayload`는 선제 발화 후보 조회와 실제 표시 직전 claim에
함께 쓰인다. `GET /child-support/proactive-prompt`는 side effect 없는 status 조회이며
새 conversation이나 message를 만들지 않는다. UI/background가 prompt를 실제로 띄울 수
있을 때 `POST /child-support/proactive-prompt/claim`을 호출하면 그때 첫 assistant turn과
agent-local `conversation_id`가 생성된다. UI는 이 값을 후속
`ChildSupportConversationRequestPayload.conversation_id`로 다시 보내기만 한다.

Wellbeing space-web은 `analysis_events.category_scores`를 사용자 화면용
`nodes`/`edges` view-model로 투영한 결과만 노출한다. category score 원본,
baseline, edge weight 알고리즘 의미는 agent service/strategy가 소유하고,
family extension은 payload를 재해석하지 않는다.
