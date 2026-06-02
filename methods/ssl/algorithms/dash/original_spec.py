"""USB Dash 원본 비교 기준."""

USB_DASH_ORIGINAL_COMMIT = "1ef4cbebcc0b368158315aeb425053858cf6c845"
USB_DASH_ORIGINAL_PATH = "semilearn/algorithms/dash/dash.py"
USB_DASH_ORIGINAL_UTILS_PATH = "semilearn/algorithms/dash/utils.py"

USB_DASH_PRESERVED_FLOW = (
    "labeled logits로 supervised CE loss를 계산한다.",
    "student model로 unlabeled weak/strong logits를 계산한다.",
    "weak logits에서 temperature soft pseudo-label 또는 hard pseudo-label을 만든다.",
    "weak logits와 pseudo-label 사이의 per-sample CE가 dynamic threshold rho 이하인 "
    "sample만 unlabeled consistency loss에 사용한다.",
    "rho = max(C * gamma^-rho_update_cnt * rho_init, rho_min) 수식을 유지한다.",
    "rho가 rho_min에 도달하면 hard pseudo-label로 전환한다.",
    "total_loss = sup_loss + lambda_u * unsup_loss 구조를 유지한다.",
)

USB_DASH_TRACE_MIND_ADAPTATIONS = (
    "USB AlgorithmBase/AMP/distributed glue는 중앙 Query SSL trainer가 소유한다.",
    "USB warm-up supervised phase 뒤 eval/loss로 rho_init을 잡는 흐름은 "
    "TraceMind 중앙 SSL의 supervised seed 시작점에 맞춰 학습 시작 전 selection "
    "loss를 주입하는 방식으로 연결한다.",
    "USB self.it와 num_iter_per_epoch 기반 10-epoch 업데이트는 QuerySslStepContext의 "
    "epoch_index/step_index로 표현한다.",
    "feature dict와 checkpoint IO는 trainer/runtime 책임이라 algorithm core에서 "
    "제외한다.",
)
