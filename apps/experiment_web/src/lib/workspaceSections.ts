import type { CatalogSectionPayload } from "../types";

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
  return sections.filter((section) => visibleSet.has(section.section_name));
}

export function getSectionDisplayCopy(
  section: CatalogSectionPayload,
  entrypointName: string | null,
): {
  displayName: string;
  description: string | null;
} {
  if (entrypointName === "train_lora_pseudo_label_classifier") {
    if (section.section_name === "lora_train_sources") {
      return {
        displayName: "적응 데이터 소스",
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
    if (section.section_name === "query_ssl_train_sources") {
      return {
        displayName: "적응 데이터 소스",
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
    if (section.section_name === "bootstrap_teacher_sources") {
      return {
        displayName: "적응 데이터 소스",
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
    if (section.section_name === "lora_train_sources") {
      return {
        displayName: "적응 데이터 소스",
        description:
          "gold labeled seed 또는 재사용 split을 사용한 지도 적응 입력입니다.",
      };
    }
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

