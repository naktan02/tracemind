import type { CatalogSectionPayload } from "../types";

export interface CentralMethodOption {
  methodId: string;
  displayName: string;
  description: string;
  entrypointName: string;
  selectedItemsBySection: Record<string, string>;
}

export type WorkspaceSectionPresentation = "cards" | "list";

const CENTRAL_ADAPTATION_COMMON_SECTION_NAMES = [
  "dataset_presets",
  "runtime_presets",
  "paper_backbones",
  "peft_methods",
  "lora_run_presets",
];

const ENTRYPOINT_VISIBLE_SECTION_NAMES: Record<string, string[] | null> = {
  train_softmax_classifier: [
    "dataset_presets",
    "embedding_presets",
    "runtime_presets",
  ],
  seed_prototypes: [
    "dataset_presets",
    "embedding_presets",
    "runtime_presets",
    "prototype_builders",
  ],
  train_lora_classifier: [
    ...CENTRAL_ADAPTATION_COMMON_SECTION_NAMES,
    "lora_train_sources",
    "initial_checkpoints",
  ],
  train_lora_pseudo_label_classifier: [
    ...CENTRAL_ADAPTATION_COMMON_SECTION_NAMES,
    "lora_train_sources",
    "pseudo_label_algorithms",
    "initial_checkpoints",
  ],
  train_lora_fixmatch: [
    ...CENTRAL_ADAPTATION_COMMON_SECTION_NAMES,
    "query_ssl_train_sources",
    "query_ssl_methods",
    "query_ssl_augmenters",
    "initial_checkpoints",
  ],
  train_lora_bootstrap_classifier_teacher: [
    ...CENTRAL_ADAPTATION_COMMON_SECTION_NAMES,
    "bootstrap_teacher_sources",
    "pseudo_label_algorithms",
    "initial_checkpoints",
  ],
  run_federated_simulation: null,
};

const ENTRYPOINT_SECTION_ORDER: Record<string, string[] | null> = {
  train_softmax_classifier: [
    "dataset_presets",
    "embedding_presets",
    "runtime_presets",
  ],
  seed_prototypes: [
    "dataset_presets",
    "embedding_presets",
    "prototype_builders",
    "runtime_presets",
  ],
  train_lora_classifier: [
    "lora_train_sources",
    "dataset_presets",
    "initial_checkpoints",
    "paper_backbones",
    "peft_methods",
    "lora_run_presets",
    "runtime_presets",
  ],
  train_lora_pseudo_label_classifier: [
    "lora_train_sources",
    "dataset_presets",
    "initial_checkpoints",
    "pseudo_label_algorithms",
    "paper_backbones",
    "peft_methods",
    "lora_run_presets",
    "runtime_presets",
  ],
  train_lora_fixmatch: [
    "query_ssl_train_sources",
    "dataset_presets",
    "initial_checkpoints",
    "query_ssl_methods",
    "query_ssl_augmenters",
    "paper_backbones",
    "peft_methods",
    "lora_run_presets",
    "runtime_presets",
  ],
  train_lora_bootstrap_classifier_teacher: [
    "bootstrap_teacher_sources",
    "dataset_presets",
    "initial_checkpoints",
    "pseudo_label_algorithms",
    "paper_backbones",
    "peft_methods",
    "lora_run_presets",
    "runtime_presets",
  ],
  run_federated_simulation: null,
};

export function getVisibleWorkspaceSections(
  entrypointName: string | null,
  sections: CatalogSectionPayload[],
): CatalogSectionPayload[] {
  if (!entrypointName) {
    return sections;
  }
  const visibleSectionNames = ENTRYPOINT_VISIBLE_SECTION_NAMES[entrypointName];
  if (visibleSectionNames === null || visibleSectionNames === undefined) {
    return sections;
  }
  const visibleSet = new Set(visibleSectionNames);
  const orderedSectionNames = ENTRYPOINT_SECTION_ORDER[entrypointName];
  const visibleSections = sections.filter((section) =>
    visibleSet.has(section.section_name),
  );
  if (orderedSectionNames === null || orderedSectionNames === undefined) {
    return visibleSections;
  }
  const orderedIndex = new Map(
    orderedSectionNames.map((sectionName, index) => [sectionName, index]),
  );
  return [...visibleSections].sort(
    (left, right) =>
      (orderedIndex.get(left.section_name) ?? Number.MAX_SAFE_INTEGER) -
      (orderedIndex.get(right.section_name) ?? Number.MAX_SAFE_INTEGER),
  );
}

const CENTRAL_METHOD_OPTIONS: CentralMethodOption[] = [
  {
    methodId: "gold_supervised",
    displayName: "Gold",
    description: "gold labeled seed train으로만 지도 적응합니다.",
    entrypointName: "train_lora_classifier",
    selectedItemsBySection: {},
  },
  {
    methodId: "margin_pseudo_label",
    displayName: "Margin",
    description: "margin 기준으로 의사라벨을 채택해 적응합니다.",
    entrypointName: "train_lora_pseudo_label_classifier",
    selectedItemsBySection: {
      pseudo_label_algorithms: "margin_threshold_v1",
    },
  },
  {
    methodId: "confidence_pseudo_label",
    displayName: "Confidence",
    description: "confidence 기준으로 의사라벨을 채택해 적응합니다.",
    entrypointName: "train_lora_pseudo_label_classifier",
    selectedItemsBySection: {
      pseudo_label_algorithms: "fixed_confidence_095",
    },
  },
  {
    methodId: "fixmatch",
    displayName: "FixMatch",
    description: "weak/strong multiview consistency로 적응합니다.",
    entrypointName: "train_lora_fixmatch",
    selectedItemsBySection: {},
  },
  {
    methodId: "teacher_bootstrap",
    displayName: "교사 부트스트랩",
    description: "teacher가 unlabeled pool을 라벨링하고 student 적응으로 잇습니다.",
    entrypointName: "train_lora_bootstrap_classifier_teacher",
    selectedItemsBySection: {},
  },
];

export function getCentralMethodOptions(): CentralMethodOption[] {
  return CENTRAL_METHOD_OPTIONS;
}

export function resolveSelectedCentralMethod(params: {
  entrypointName: string | null;
  selectedItemNameBySection: Record<string, string | null>;
}): CentralMethodOption | null {
  const { entrypointName, selectedItemNameBySection } = params;
  if (!entrypointName) {
    return null;
  }
  for (const option of CENTRAL_METHOD_OPTIONS) {
    if (option.entrypointName !== entrypointName) {
      continue;
    }
    const matches = Object.entries(option.selectedItemsBySection).every(
      ([sectionName, itemName]) =>
        selectedItemNameBySection[sectionName] === itemName,
    );
    if (matches) {
      return option;
    }
  }
  return CENTRAL_METHOD_OPTIONS.find(
    (option) => option.entrypointName === entrypointName,
  ) ?? null;
}

export function getSectionDisplayCopy(
  section: CatalogSectionPayload,
  entrypointName: string | null,
): {
  displayName: string;
  description: string | null;
} {
  if (entrypointName === "train_lora_pseudo_label_classifier") {
    if (section.section_name === "dataset_presets") {
      return {
        displayName: "평가 데이터",
        description:
          "이 실험에서 검증/테스트에 사용할 dataset alias를 고릅니다.",
      };
    }
    if (section.section_name === "lora_train_sources") {
      return {
        displayName: "학생 데이터",
        description:
          "gold labeled seed 또는 재사용 split 같은 지도 데이터 소스를 고릅니다.",
      };
    }
    if (section.section_name === "pseudo_label_algorithms") {
      return {
        displayName: "방법론",
        description:
          "의사라벨을 어떤 규칙으로 채택할지 고릅니다. 같은 자리에서 다른 적응 방법과 비교합니다.",
      };
    }
  }

  if (entrypointName === "train_lora_fixmatch") {
    if (section.section_name === "dataset_presets") {
      return {
        displayName: "평가 데이터",
        description:
          "이 실험에서 검증/테스트에 사용할 dataset alias를 고릅니다.",
      };
    }
    if (section.section_name === "query_ssl_train_sources") {
      return {
        displayName: "학생 데이터",
        description:
          "labeled train과 unlabeled pool이 함께 준비된 SSL 입력 소스를 고릅니다.",
      };
    }
    if (section.section_name === "query_ssl_methods") {
      return {
        displayName: "방법론",
        description:
          "FixMatch 같은 consistency 목표함수를 고릅니다. 의사라벨 방법론과 같은 비교 위치에 둡니다.",
      };
    }
  }

  if (entrypointName === "train_lora_bootstrap_classifier_teacher") {
    if (section.section_name === "dataset_presets") {
      return {
        displayName: "평가 데이터",
        description:
          "student 적응 결과를 검증/테스트할 dataset alias를 고릅니다.",
      };
    }
    if (section.section_name === "bootstrap_teacher_sources") {
      return {
        displayName: "교사 데이터",
        description:
          "teacher seed train과 unlabeled pool을 어디서 가져올지 고릅니다.",
      };
    }
    if (section.section_name === "pseudo_label_algorithms") {
      return {
        displayName: "방법론",
        description:
          "teacher가 unlabeled 예시를 어떤 규칙으로 채택할지 고릅니다.",
      };
    }
  }

  if (entrypointName === "train_lora_classifier") {
    if (section.section_name === "dataset_presets") {
      return {
        displayName: "평가 데이터",
        description:
          "이 실험에서 검증/테스트에 사용할 dataset alias를 고릅니다.",
      };
    }
    if (section.section_name === "lora_train_sources") {
      return {
        displayName: "학생 데이터",
        description:
          "gold labeled seed 또는 재사용 split을 사용한 지도 적응 입력입니다.",
      };
    }
  }

  if (
    entrypointName?.startsWith("train_lora_") &&
    section.section_name === "initial_checkpoints"
  ) {
    return {
      displayName: "초기 체크포인트",
      description:
        "fresh start, 고정 분류기 seed, 기존 LoRA seed 중 어디서 시작할지 고릅니다.",
    };
  }

  if (entrypointName === "seed_prototypes") {
    if (section.section_name === "prototype_builders") {
      return {
        displayName: "프로토타입 자산 방식",
        description:
          "프로토타입 pack을 어떤 방식으로 만들지 고릅니다. 분류기 기준선과는 별도 자산입니다.",
      };
    }
  }

  return {
    displayName: section.display_name,
    description: section.description,
  };
}

export function getSectionPresentation(
  section: CatalogSectionPayload,
  entrypointName: string | null,
): WorkspaceSectionPresentation {
  if (!entrypointName?.startsWith("train_lora_")) {
    return "cards";
  }
  const listStyleSections = new Set([
    "dataset_presets",
    "lora_train_sources",
    "query_ssl_train_sources",
    "bootstrap_teacher_sources",
    "initial_checkpoints",
    "pseudo_label_algorithms",
    "query_ssl_methods",
    "query_ssl_augmenters",
  ]);
  return listStyleSections.has(section.section_name) ? "list" : "cards";
}

export function getEntrypointGuide(entrypointName: string | null): string | null {
  const guides: Record<string, string> = {
    train_softmax_classifier:
      "라벨된 train 데이터로 초기 고정 분류기 기준선을 만듭니다. validation/test는 평가와 선택 기준에 사용됩니다.",
    seed_prototypes:
      "train split에서 프로토타입 자산을 만듭니다. 평가용 validation/test는 생성 단계가 아니라 별도 prototype 평가에서 씁니다.",
    train_lora_classifier:
      "gold labeled train source로 지도 적응을 수행합니다. 가장 단순한 중앙 적응 비교선입니다.",
    train_lora_pseudo_label_classifier:
      "의사라벨 채택 규칙으로 새 query-derived rows를 골라 지속 적응을 수행합니다.",
    train_lora_fixmatch:
      "weak/strong multiview와 consistency loss를 쓰는 FixMatch 계열 중앙 적응입니다.",
    train_lora_bootstrap_classifier_teacher:
      "teacher가 unlabeled pool에서 pseudo-label을 만들고, 그 결과를 student 적응에 연결합니다.",
    run_federated_simulation:
      "공통 FL 실행 파이프라인 위에서 family, aggregation, objective 조합을 바꿔 비교합니다.",
  };
  return entrypointName ? guides[entrypointName] ?? null : null;
}
