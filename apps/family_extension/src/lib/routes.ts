export const APP_ROUTES = ["/child", "/unlock", "/parent"] as const;

export type AppRoute = (typeof APP_ROUTES)[number];

export type AppRouteMeta = {
  readonly label: string;
  readonly eyebrow: string;
  readonly title: string;
  readonly description: string;
};

export const ROUTE_META: Record<AppRoute, AppRouteMeta> = {
  "/child": {
    label: "아이용 화면",
    eyebrow: "Child View",
    title: "아이용 wellbeing summary",
    description:
      "아이에게 보여줄 현재 상태, 짧은 요약, 행동 제안을 wellbeing summary 한 건으로 보여줍니다.",
  },
  "/unlock": {
    label: "부모 잠금 화면",
    eyebrow: "Parent Unlock",
    title: "부모용 PIN 진입 shell",
    description:
      "부모용 상세 화면으로 들어가기 전 PIN 검증 흐름이 연결될 자리입니다. 실제 unlock API 연결은 다음 단계에서 붙입니다.",
  },
  "/parent": {
    label: "부모 상세 화면",
    eyebrow: "Parent Detail",
    title: "부모용 detail shell",
    description:
      "전체 wellbeing signal, 최근 추이, 권장 행동이 올라갈 보호자용 상세 화면 자리입니다.",
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
