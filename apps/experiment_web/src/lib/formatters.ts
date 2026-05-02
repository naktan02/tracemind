export function formatMetadataValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "null";
  }
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return String(value);
  }
  return JSON.stringify(value);
}

export function formatScalarValue(value: string | number | boolean): string {
  return String(value);
}

export function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function asErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export function formatMetricKey(metricKey: string): string {
  const knownLabels: Record<string, string> = {
    "validation.accuracy_top_1": "검증 ACC",
    "validation.macro_f1": "검증 Macro-F1",
    "validation.weighted_f1": "검증 Weighted-F1",
    "test.accuracy_top_1": "테스트 ACC",
    "test.macro_f1": "테스트 Macro-F1",
    "selection.accuracy_top_1": "선택 정답률",
    "teacher.accepted_ratio": "교사 채택 비율",
    "teacher.accepted_hidden_label_accuracy": "교사 채택 정답률",
  };
  return knownLabels[metricKey] ?? humanizeIdentifier(metricKey);
}

export function formatMetricValue(value: number): string {
  if (Number.isInteger(value)) {
    return String(value);
  }
  if (Math.abs(value) >= 1) {
    return value.toFixed(3);
  }
  return value.toFixed(4);
}

export function formatTrackName(trackName: string): string {
  const knownLabels: Record<string, string> = {
    seed: "기준선 생성",
    central_adaptation: "중앙 적응 비교",
    federated_runtime: "연합 런타임 점검",
  };
  return knownLabels[trackName] ?? humanizeIdentifier(trackName);
}

export function formatEntrypointName(entrypointName: string): string {
  const knownLabels: Record<string, string> = {
    train_softmax_classifier: "고정 분류기 학습",
    seed_prototypes: "프로토타입 자산 생성",
    train_lora_classifier: "Gold 지도 적응",
    train_lora_pseudo_label_classifier: "의사라벨 적응 실행",
    train_lora_fixmatch: "FixMatch 적응 실행",
    train_lora_bootstrap_classifier_teacher: "교사 부트스트랩 적응 실행",
    run_federated_simulation: "공통 FL 실행",
  };
  return knownLabels[entrypointName] ?? humanizeIdentifier(entrypointName);
}

export function formatSectionName(sectionName: string): string {
  const knownLabels: Record<string, string> = {
    dataset_presets: "데이터셋",
    embedding_presets: "임베딩",
    runtime_presets: "실행 환경",
    prototype_builders: "프로토타입 빌더",
    paper_backbones: "백본",
    peft_methods: "PEFT 방법",
    lora_run_presets: "적응 실행 프리셋",
    lora_train_sources: "지도 적응 데이터 소스",
    query_ssl_train_sources: "SSL 적응 데이터 소스",
    bootstrap_teacher_sources: "교사 초기 데이터 소스",
    pseudo_label_algorithms: "의사라벨 선택 방식",
    query_ssl_methods: "SSL 목표 함수",
    query_ssl_augmenters: "멀티뷰 증강",
    initial_checkpoints: "초기 체크포인트",
    federated_run_presets: "연합 실행 프리셋",
    training_algorithm_profiles: "학습 알고리즘 프로필",
    adapter_families: "어댑터 패밀리",
    aggregation_backends: "집계 방식",
    scoring_backends: "점수 계산 방식",
    training_backends: "로컬 학습 백엔드",
    example_generation_backends: "예제 생성 방식",
    privacy_guards: "보호 장치",
  };
  return knownLabels[sectionName] ?? humanizeIdentifier(sectionName);
}

export function formatCompileSupport(value: string): string {
  const knownLabels: Record<string, string> = {
    entrypoint: "실행 가능",
    preset_selector: "선택 가능",
    metadata_only: "참고용",
  };
  return knownLabels[value] ?? humanizeIdentifier(value);
}

export function humanizeIdentifier(value: string): string {
  return value
    .replace(/[._]/g, " ")
    .replace(/\b\w/g, (letter: string) => letter.toUpperCase());
}

export function formatRunStatus(status: string): string {
  const knownLabels: Record<string, string> = {
    running: "실행 중",
    succeeded: "성공",
    failed: "실패",
    interrupted: "중단",
  };
  return knownLabels[status] ?? humanizeIdentifier(status);
}
