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
    eyebrow: "처음 설정",
    title: "아이/부모 PIN 초기 설정",
    description:
      "아이 화면과 보호자 화면을 안전하게 나누기 위해 각각의 PIN을 설정합니다.",
  },
  "/gate": {
    label: "역할 선택",
    eyebrow: "화면 선택",
    title: "들어갈 화면 선택",
    description:
      "아이용 자기 점검 화면과 보호자 안내 화면 중 하나를 선택합니다.",
  },
  "/child/unlock": {
    label: "아이용 PIN",
    eyebrow: "아이용 PIN",
    title: "아이용 PIN 검증",
    description:
      "내 마음 상태와 AI 마음 도움을 확인하기 전에 PIN을 입력합니다.",
  },
  "/parent/unlock": {
    label: "부모용 PIN",
    eyebrow: "부모용 PIN",
    title: "부모용 PIN 검증",
    description:
      "보호자 안내 화면으로 들어가기 전에 PIN을 입력합니다.",
  },
  "/child": {
    label: "AI 마음 도움",
    eyebrow: "아이 페이지",
    title: "내 마음 상태와 AI 도움",
    description:
      "상태 단계, 위험도 변화, 개인 맥락 기반 대화 도움을 한곳에서 확인합니다.",
  },
  "/parent": {
    label: "보호자 안내",
    eyebrow: "부모 페이지",
    title: "보호자 대응 안내",
    description:
      "원문과 그래프 없이 현재 필요한 관심 수준과 대응 방향만 안내합니다.",
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
