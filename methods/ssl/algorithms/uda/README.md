# UDA

`uda.py`는 Microsoft USB commit
`1ef4cbebcc0b368158315aeb425053858cf6c845`의
`semilearn/algorithms/uda/uda.py` 핵심 train step을 TraceMind Query SSL
surface로 옮긴 구현이다.

보존한 핵심 흐름:

1. labeled logits와 unlabeled weak/strong logits를 계산한다.
2. TSA schedule로 labeled CE sample mask를 만든다.
3. weak probability의 max confidence로 fixed threshold mask를 만든다.
4. weak probability를 soft pseudo-label target으로 사용한다.
5. strong logits에 masked soft CE consistency loss를 적용한다.
6. `total_loss = sup_loss + lambda_u * unsup_loss`를 계산한다.

주의할 점:

- USB 원본은 `PseudoLabelingHook(..., softmax=False)`에 이미 softmax된 weak
  probability를 넘긴다. 따라서 기본 `T=0.4` 인자는 이 경로에서 soft target을 다시
  sharpen하지 않는다. TraceMind도 이 동작을 그대로 보존한다.
- USB 원본에는 CReST 같은 imbalanced 확장용 optional `DistAlignHook` 호출 경로가
  있지만, `uda_usb_v1` 기본 preset에서는 해당 hook을 등록하지 않는다.
- EMA teacher, distribution alignment, projection head, memory queue는 UDA 핵심이
  아니므로 추가하지 않는다.
