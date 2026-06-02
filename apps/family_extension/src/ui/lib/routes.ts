import type { FamilyAccessRole } from "../../contracts/generated";

export const APP_ROUTES = [
  "/setup",
  "/gate",
  "/child/unlock",
  "/parent/unlock",
  "/child",
  "/parent",
] as const;

export type AppRoute = (typeof APP_ROUTES)[number];

export type AppRouteMeta = {
  readonly label: string;
  readonly eyebrow: string;
  readonly title: string;
  readonly description: string;
};

export const ROUTE_META: Record<AppRoute, AppRouteMeta> = {
  "/setup": {
    label: "초기 설정",
    eyebrow: "First Run Setup",
    title: "아이/부모 PIN 초기 설정",
    description:
      "이 PC의 로컬 agent만 사용하는 기본 보호 정책과 role별 PIN을 먼저 설정합니다.",
  },
  "/gate": {
    label: "역할 선택",
    eyebrow: "Family Gate",
    title: "들어갈 화면 선택",
    description:
      "child/parent 중 어떤 보호 화면으로 들어갈지 먼저 고르고 각 role의 PIN 검증으로 이동합니다.",
  },
  "/child/unlock": {
    label: "아이용 PIN",
    eyebrow: "Child Unlock",
    title: "아이용 PIN 검증",
    description:
      "아이용 현재 상태 화면으로 들어가기 전에 role 전용 PIN 검증을 수행합니다.",
  },
  "/parent/unlock": {
    label: "부모용 PIN",
    eyebrow: "Parent Unlock",
    title: "부모용 PIN 검증",
    description:
      "부모용 상세 화면으로 들어가기 전에 role 전용 PIN 검증을 수행합니다.",
  },
  "/child": {
    label: "AI 마음 도움",
    eyebrow: "Child Support",
    title: "아이용 AI 마음 도움",
    description:
      "아이에게 보여줄 현재 상태와 로컬 agent의 안전한 대화 도움을 함께 보여줍니다.",
  },
  "/parent": {
    label: "부모 상세 화면",
    eyebrow: "Parent Detail",
    title: "부모용 wellbeing detail",
    description:
      "현재 상태, 최근 추이, 권장 행동을 보호자용 상세 화면에서 보여줍니다.",
  },
};

export function isAppRoute(value: string): value is AppRoute {
  return (APP_ROUTES as readonly string[]).includes(value);
}

export function normalizeRoute(
  rawRoute: string | null | undefined,
  fallback: AppRoute,
): AppRoute {
  if (rawRoute == null || rawRoute.trim() === "") {
    return fallback;
  }
  return isAppRoute(rawRoute) ? rawRoute : fallback;
}

type RouteAccessOptions = {
  activeRole: FamilyAccessRole | null;
  isSetupComplete: boolean;
};

export function isRouteLocked(
  route: AppRoute,
  options: RouteAccessOptions,
): boolean {
  return resolveAccessibleRoute(route, options) !== route;
}

export function resolveAccessibleRoute(
  route: AppRoute,
  options: RouteAccessOptions,
): AppRoute {
  if (!options.isSetupComplete) {
    return "/setup";
  }

  if (options.activeRole == null) {
    if (route === "/child" || route === "/parent" || route === "/setup") {
      return "/gate";
    }
    return route;
  }

  if (options.activeRole === "child") {
    return route === "/child" ? "/child" : "/child";
  }

  return route === "/parent" ? "/parent" : "/parent";
}
