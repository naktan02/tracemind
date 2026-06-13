# Wellbeing Services

`wellbeing/`은 family extension과 child support가 읽는 agent-local wellbeing 출력을
소유한다.

읽기 시작점:

- `signal/`: local inference 결과를 wellbeing summary/timeseries payload로 투영한다.
- `space_web/`: category score coactivation을 아이용 공간웹 payload로 투영한다.
- `family_access/`: child/parent setup, unlock, legacy parent auth adapter를 소유한다.
- `child_support/`: 아이용 지원 대화 context, safety routing, response policy, LLM adapter를 소유한다.

raw text, conversation state, PIN state 같은 개인 상태는 agent-local repository 경계에
남기고, UI는 `agent/src/contracts/`의 payload만 소비한다.
