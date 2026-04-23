import { useEffect, useMemo, useState } from "react";

import { checkLocalProgramHealth } from "./api/wellbeing";
import { getAgentApiBaseUrl } from "./api/client";
import { ChildPage } from "./pages/child/ChildPage";
import { ParentPage } from "./pages/parent/ParentPage";
import { UnlockPage } from "./pages/unlock/UnlockPage";
import { AppRoute, ROUTE_META, normalizeRoute } from "./lib/routes";

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
  const [healthState, setHealthState] = useState<
    "checking" | "connected" | "offline"
  >("checking");
  const [pin, setPin] = useState("");

  useEffect(() => {
    const onHashChange = () => {
      setCurrentRoute(normalizeRoute(getHashRoute(), fallbackRoute));
    };

    updateHash(currentRoute);
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [currentRoute, fallbackRoute]);

  useEffect(() => {
    let cancelled = false;

    async function loadHealth() {
      const isConnected = await checkLocalProgramHealth();
      if (!cancelled) {
        setHealthState(isConnected ? "connected" : "offline");
      }
    }

    void loadHealth();
    return () => {
      cancelled = true;
    };
  }, []);

  const routeMeta = useMemo(() => ROUTE_META[currentRoute], [currentRoute]);

  function moveTo(route: AppRoute) {
    setCurrentRoute(route);
    updateHash(route);
  }

  return (
    <div className="app-shell">
      <aside className="side-rail">
        <div className="brand-block">
          <p className="brand-eyebrow">TraceMind Family</p>
          <h1>Family Extension</h1>
          <p className="brand-copy">
            child popup과 parent detail이 같은 output contract를 소비하도록 여는
            shell 단계입니다.
          </p>
        </div>

        <nav className="route-nav" aria-label="family extension routes">
          {(
            Object.entries(ROUTE_META) as [AppRoute, (typeof ROUTE_META)[AppRoute]][]
          ).map(([route, meta]) => (
            <button
              key={route}
              className={route === currentRoute ? "route-link active" : "route-link"}
              type="button"
              onClick={() => moveTo(route)}
            >
              <span>{meta.label}</span>
              <small>{route}</small>
            </button>
          ))}
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
            <span className="badge">Phase 4 shell</span>
            <span className="badge subtle">wellbeing_signal consumer</span>
          </div>
        </header>

        {currentRoute === "/child" && (
          <ChildPage
            onMoveToUnlock={() => moveTo("/unlock")}
            onMoveToParent={() => moveTo("/parent")}
          />
        )}
        {currentRoute === "/unlock" && (
          <UnlockPage
            pin={pin}
            onPinChange={setPin}
            onMoveToChild={() => moveTo("/child")}
            onMoveToParent={() => moveTo("/parent")}
          />
        )}
        {currentRoute === "/parent" && (
          <ParentPage
            onMoveToChild={() => moveTo("/child")}
            onMoveToUnlock={() => moveTo("/unlock")}
          />
        )}
      </main>
    </div>
  );
}
