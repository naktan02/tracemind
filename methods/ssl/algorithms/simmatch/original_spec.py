"""USB SimMatch 원본 비교 기준."""

USB_SIMMATCH_ORIGINAL_COMMIT = "1ef4cbebcc0b368158315aeb425053858cf6c845"
USB_SIMMATCH_ORIGINAL_PATH = "semilearn/algorithms/simmatch/simmatch.py"

USB_SIMMATCH_PRESERVED_FLOW = (
    "base classifier feature를 projection head로 L2-normalize한다.",
    "labeled feature memory bank와 labels bank를 유지한다.",
    "weak logits probability에 queue distribution alignment를 적용한다.",
    "weak projected feature와 memory bank similarity로 teacher probability를 만든다.",
    "teacher similarity probability에 bank label별 weak class probability factor를 "
    "곱해 정규화한다.",
    "strong projected feature와 memory bank similarity로 student probability를 만들고 "
    "feature-level in_loss를 계산한다.",
    "smoothing_alpha가 1보다 작으면 similarity로 aggregate한 class probability와 "
    "weak probability를 섞는다.",
    "고정 confidence threshold mask와 CE consistency loss를 사용한다.",
    "labeled projected feature와 label/index로 memory bank를 갱신한다.",
)

USB_SIMMATCH_TRACE_MIND_ADAPTATIONS = (
    "USB AlgorithmBase/AMP/distributed glue는 중앙 Query SSL trainer가 소유한다.",
    "USB가 텍스트 dataset에서 use_ema_teacher=False로 전환하는 경로를 기본값으로 "
    "사용한다.",
    "USB idx_lb는 labeled dataloader의 stable row_indices로 표현한다.",
    "USB K=args.lb_dest_len은 labeled_row_count lifecycle capability로 설정한다.",
    "SimMatch projection head는 CoMatch와 공통 `SslProjectionHead`를 사용한다.",
    "feature dict와 checkpoint IO는 trainer/runtime 책임이라 algorithm core에서 "
    "제외한다.",
)
