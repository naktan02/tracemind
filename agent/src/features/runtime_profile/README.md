# Runtime Profile

`runtime_profile/`은 서버가 배포한 agent 상시 분석 runtime profile을
agent-local SQLite에 저장하고 active profile을 조회하는 feature다.

역할:

- 서버 runtime profile payload를 agent-local DB에 저장한다.
- active profile은 최대 하나만 허용한다.
- 날짜(`received_at`, `activated_at`, `server_validated_at`)는 표시, 감사,
  stale 정책 판단에만 쓰고 최신성 비교에는 쓰지 않는다.
- 최신성 비교는 `profile_id`, `profile_revision`, `payload_checksum`을 사용한다.
- sync service는 서버 `/api/v1/agent-runtime-profile/current|validate`를 호출하고,
  새 profile이 있으면 shared adapter current state도 함께 pull한 뒤 active profile로
  저장한다.
- inference pipeline은 active profile이 있으면 profile의 scorer/embedding/runtime
  선택값을 우선 사용한다. active profile이 없으면 기존 env 기반 pipeline 설정을 쓴다.
- training task 실행은 active profile이 있을 때만 task `model_revision`과
  `required_state_kind` 정합성을 검사한다. profile이 비어 있으면 기존처럼 통과한다.

이 feature는 captured text 저장소가 아니다. captured text 분석, typing segment
분석, wellbeing projection, training task 정합성 검증이 함께 읽는 agent runtime
설정이다.
