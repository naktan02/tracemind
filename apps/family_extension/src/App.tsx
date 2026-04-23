import { useEffect, useMemo, useState } from "react";

import { ConnectionStateBanner } from "./components/ConnectionStateBanner";
import { getAgentApiBaseUrl } from "./api/client";
import { useLocalProgramHealth } from "./hooks/useLocalProgramHealth";
import { useParentUnlock } from "./hooks/useParentUnlock";
import { ChildPage } from "./pages/child/ChildPage";
import { ParentPage } from "./pages/parent/ParentPage";
import { UnlockPage } from "./pages/unlock/UnlockPage";
import {
  AppRoute,
  ROUTE_META,
  isRouteLocked,
  normalizeRoute,
  resolveAccessibleRoute,
} from "./lib/routes";

type AppProps = {
  initialRoute?: string;
};

function getHashRoute(): string | null {
  const rawHash = window.location.hash.replace(/^#/, "");
  return rawHash === "" ? null : rawHash;
}

function updateHash(route: AppRoute) {
  if (window.location.hash !== `#${route}`) {
    window.location.hash = route;
  }
}

export default function App({ initialRoute }: AppProps) {
  const fallbackRoute = normalizeRoute(initialRoute, "/child");
  const [currentRoute, setCurrentRoute] = useState<AppRoute>(() =>
    normalizeRoute(getHashRoute(), fallbackRoute),
  );
  const healthState = useLocalProgramHealth();
  const [pin, setPin] = useState("");
  const {
    activeSessionExpiresAt,
    clearParentSession,
    hasActiveParentSession,
    submitUnlock,
    unlockState,
  } = useParentUnlock();

  useEffect(() => {
    const onHashChange = () => {
      const nextRoute = resolveAccessibleRoute(
        normalizeRoute(getHashRoute(), fallbackRoute),
        { hasActiveParentSession },
      );
      setCurrentRoute(nextRoute);
      updateHash(nextRoute);
    };

    updateHash(currentRoute);
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [currentRoute, fallbackRoute, hasActiveParentSession]);

  const routeMeta = useMemo(() => ROUTE_META[currentRoute], [currentRoute]);
  const visibleRoutes = useMemo<AppRoute[]>(
    () => (hasActiveParentSession ? ["/child", "/parent"] : ["/child", "/unlock"]),
    [hasActiveParentSession],
  );

  useEffect(() => {
    if (currentRoute === "/parent" && !hasActiveParentSession) {
      setCurrentRoute("/unlock");
      updateHash("/unlock");
    }
  }, [currentRoute, hasActiveParentSession]);

  function moveTo(route: AppRoute) {
    const nextHasActiveParentSession =
      currentRoute === "/parent" && route !== "/parent"
        ? false
        : hasActiveParentSession;
    if (!nextHasActiveParentSession && currentRoute === "/parent") {
      clearParentSession();
    }
    const nextRoute = resolveAccessibleRoute(route, {
      hasActiveParentSession: nextHasActiveParentSession,
    });
    setCurrentRoute(nextRoute);
    updateHash(nextRoute);
  }

  async function handleUnlockSubmit() {
    const response = await submitUnlock(pin);
    if (response?.granted) {
      setPin("");
      moveTo("/parent");
    }
  }

  return (
    <div className="app-shell">
      <aside className="side-rail">
        <div className="brand-block">
          <p className="brand-eyebrow">TraceMind Family</p>
          <h1>Family Extension</h1>
          <p className="brand-copy">
            child popup은 이제 wellbeing summary를 읽고, parent detail은 다음
            단계에서 같은 output contract를 이어받습니다.
          </p>
        </div>

        <nav className="route-nav" aria-label="family extension routes">
          {(visibleRoutes as AppRoute[]).map((route) => {
            const meta = ROUTE_META[route];
            const locked = isRouteLocked(route, { hasActiveParentSession });
            return (
              <button
                key={route}
                className={
                  route === currentRoute
                    ? "route-link active"
                    : locked
                      ? "route-link locked"
                      : "route-link"
                }
                type="button"
                onClick={() => moveTo(route)}
              >
                <span>{meta.label}</span>
                <small>{locked ? `${route} · PIN 필요` : route}</small>
              </button>
            );
          })}
        </nav>

        <div className="status-panel">
          <p className="status-label">로컬 프로그램 연결</p>
          <strong
            className={
              healthState === "connected"
                ? "status-text success"
                : healthState === "offline"
                  ? "status-text warning"
                  : "status-text"
            }
          >
            {healthState === "checking" && "확인 중"}
            {healthState === "connected" && "연결됨"}
            {healthState === "offline" && "연결 안 됨"}
          </strong>
          <span className="status-hint">{getAgentApiBaseUrl()}</span>
        </div>
      </aside>

      <main className="main-panel">
        <header className="page-header">
          <div>
            <p className="page-eyebrow">{routeMeta.eyebrow}</p>
            <h2>{routeMeta.title}</h2>
            <p className="page-description">{routeMeta.description}</p>
          </div>
          <div className="badge-row">
            <span className="badge">Phase 9 runtime projection</span>
            <span className="badge subtle">wellbeing_signal consumer</span>
          </div>
        </header>
        <ConnectionStateBanner healthState={healthState} />

        {currentRoute === "/child" && (
          <ChildPage onMoveToUnlock={() => moveTo("/unlock")} />
        )}
        {currentRoute === "/unlock" && (
          <UnlockPage
            pin={pin}
            unlockState={unlockState}
            onPinChange={setPin}
            onSubmitUnlock={handleUnlockSubmit}
            onMoveToChild={() => moveTo("/child")}
            onMoveToParent={() => moveTo("/parent")}
          />
        )}
        {currentRoute === "/parent" && (
          <ParentPage
            activeSessionExpiresAt={activeSessionExpiresAt}
            hasActiveParentSession={hasActiveParentSession}
            onMoveToChild={() => moveTo("/child")}
            onMoveToUnlock={() => moveTo("/unlock")}
          />
        )}
      </main>
    </div>
  );
}
