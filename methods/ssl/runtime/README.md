# Query SSL Runtime

이 폴더는 특정 SSL algorithm 수식이 아니라 training loop lifecycle에 붙는 재사용
runtime primitive를 둔다.

- `lifecycle.py`: trainer가 algorithm optional hook을 호출하는 lifecycle dispatch
- `ema.py`: MeanTeacher 같은 teacher-student method가 쓰는 EMA weight source
- `schedules.py`: USB 계열 method가 공유하는 step 기반 warm-up 수식

Algorithm package는 여기의 primitive를 호출하되, method별 loss와 state 의미는
`methods/ssl/algorithms/<method>/`가 계속 소유한다.
