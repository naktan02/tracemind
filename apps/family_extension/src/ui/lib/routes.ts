import type { FamilyAccessRole } from "../../contracts/generated";

export const APP_ROUTES = [
  "/setup",
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
    title: "본인/부모 PIN 초기 설정",
    description:
      "본인 페이지와 부모 페이지를 안전하게 나누기 위해 각각의 PIN을 설정합니다.",
  },
  "/child/unlock": {
    label: "본인 PIN",
    eyebrow: "본인 PIN",
    title: "본인 PIN 확인",
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
    eyebrow: "본인 페이지",
    title: "내 마음 상태",
    description: "오늘의 상태를 확인하고 필요한 도움을 받아보세요.",
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

export function resolveAccessibleRoute(
  route: AppRoute,
  options: RouteAccessOptions,
): AppRoute {
  if (!options.isSetupComplete) {
    return "/setup";
  }

  if (options.activeRole == null) {
    if (route === "/child") {
      return "/child/unlock";
    }
    if (route === "/parent") {
      return "/parent/unlock";
    }
    if (route === "/setup") {
      return "/child/unlock";
    }
    return route;
  }

  if (options.activeRole === "child") {
    return route === "/child" ? "/child" : "/child";
  }

  return route === "/parent" ? "/parent" : "/parent";
}
