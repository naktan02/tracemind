import { useEffect, useMemo, useState } from "react";

import { ConnectionStateBanner } from "./components/ConnectionStateBanner";
import { getAgentApiBaseUrl } from "../common/agentClient";
import { useFamilyAccess } from "./hooks/useFamilyAccess";
import { useLocalProgramHealth } from "./hooks/useLocalProgramHealth";
import { GatePage } from "./pages/gate/GatePage";
import { ChildPage } from "./pages/child/ChildPage";
import { ParentPage } from "./pages/parent/ParentPage";
import { SetupPage } from "./pages/setup/SetupPage";
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
  const fallbackRoute = normalizeRoute(initialRoute, "/gate");
  const [currentRoute, setCurrentRoute] = useState<AppRoute>(() =>
    normalizeRoute(getHashRoute(), fallbackRoute),
  );
  const healthState = useLocalProgramHealth();
  const {
    activeRole,
    activeSession,
    clearRoleSession,
    getUnlockState,
    reloadSetupStatus,
    setupStatusState,
    setupSubmissionState,
    submitRoleUnlock,
    submitSetup,
  } = useFamilyAccess();
  const [childUnlockPin, setChildUnlockPin] = useState("");
  const [parentUnlockPin, setParentUnlockPin] = useState("");
  const isSetupResolved = setupStatusState.phase === "loaded";
  const isSetupComplete = setupStatusState.status?.is_setup_complete ?? false;

  useEffect(() => {
    const onHashChange = () => {
      const nextRoute = resolveAccessibleRoute(
        normalizeRoute(getHashRoute(), fallbackRoute),
        { activeRole, isSetupComplete },
      );
      if (!isSetupResolved) {
        setCurrentRoute(normalizeRoute(getHashRoute(), fallbackRoute));
        return;
      }
      setCurrentRoute(nextRoute);
      updateHash(nextRoute);
    };

    updateHash(currentRoute);
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [activeRole, currentRoute, fallbackRoute, isSetupComplete, isSetupResolved]);

  const routeMeta = useMemo(() => ROUTE_META[currentRoute], [currentRoute]);
  const visibleRoutes = useMemo<AppRoute[]>(
    () => {
      if (!isSetupResolved) {
        return [];
      }
      if (!isSetupComplete) {
        return ["/setup"];
      }
      if (activeRole === null) {
        return ["/gate"];
      }
      return activeRole === "child" ? ["/child"] : ["/parent"];
    },
    [activeRole, isSetupComplete, isSetupResolved],
  );

  useEffect(() => {
    if (!isSetupResolved) {
      return;
    }
    const nextRoute = resolveAccessibleRoute(currentRoute, {
      activeRole,
      isSetupComplete,
    });
    if (nextRoute !== currentRoute) {
      setCurrentRoute(nextRoute);
      updateHash(nextRoute);
    }
  }, [activeRole, currentRoute, isSetupComplete, isSetupResolved]);

  function moveTo(route: AppRoute) {
    const isLeavingProtectedRoute =
      (currentRoute === "/child" && route !== "/child") ||
      (currentRoute === "/parent" && route !== "/parent");
    if (isLeavingProtectedRoute) {
      clearRoleSession();
    }
    const nextRoute = resolveAccessibleRoute(route, {
      activeRole: isLeavingProtectedRoute ? null : activeRole,
      isSetupComplete,
    });
    setCurrentRoute(nextRoute);
    updateHash(nextRoute);
  }

  async function handleRoleUnlockSubmit(role: "child" | "parent") {
    const pin = role === "child" ? childUnlockPin : parentUnlockPin;
    const response = await submitRoleUnlock(role, pin);
    if (response?.granted) {
      if (role === "child") {
        setChildUnlockPin("");
        setCurrentRoute("/child");
        updateHash("/child");
        return;
      }
      setParentUnlockPin("");
      setCurrentRoute("/parent");
      updateHash("/parent");
    }
  }

  async function handleSetupSubmit(childPin: string, parentPin: string) {
    const response = await submitSetup(childPin, parentPin);
    if (response?.is_setup_complete) {
      setCurrentRoute("/gate");
      updateHash("/gate");
    }
  }

  return (
    <div className="app-shell">
      <aside className="side-rail">
        <div className="brand-block">
          <p className="brand-eyebrow">TraceMind Family</p>
          <h1>Family Extension</h1>
          <p className="brand-copy">
            이 확장 프로그램은 이 PC의 로컬 agent만 사용합니다. setup이 끝나면
            child와 parent는 각각 잠금 해제를 거쳐 role별 화면으로 들어갑니다.
          </p>
        </div>

        <nav className="route-nav" aria-label="family extension routes">
          {(visibleRoutes as AppRoute[]).map((route) => {
            const meta = ROUTE_META[route];
            const locked = isRouteLocked(route, { activeRole, isSetupComplete });
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
            <span className="badge">Phase 10 app-level gate</span>
            <span className="badge subtle">local-only family access</span>
          </div>
        </header>
        <ConnectionStateBanner healthState={healthState} />

        {setupStatusState.phase === "loading" && (
          <section className="hero-card gate-hero">
            <div>
              <p className="eyebrow">Family Access</p>
              <h2>초기 설정 상태를 확인하는 중</h2>
              <p className="section-copy">
                이 PC의 로컬 agent가 이미 초기 설정을 마쳤는지 확인하고
                있습니다.
              </p>
            </div>
            <div className="hero-meter">
              <span className="hero-meter-label">setup status</span>
              <strong>확인 중...</strong>
            </div>
          </section>
        )}
        {setupStatusState.phase === "error" && (
          <section className="hero-card gate-hero">
            <div>
              <p className="eyebrow">Family Access</p>
              <h2>초기 설정 상태를 아직 확인하지 못했습니다</h2>
              <p className="section-copy">{setupStatusState.errorMessage}</p>
            </div>
            <div className="button-row">
              <button
                className="primary-button"
                type="button"
                onClick={() => void reloadSetupStatus()}
              >
                다시 시도
              </button>
            </div>
          </section>
        )}

        {setupStatusState.phase === "loaded" && currentRoute === "/setup" && (
          <SetupPage
            errorMessage={setupSubmissionState.errorMessage}
            isSubmitting={setupSubmissionState.phase === "submitting"}
            onSubmitSetup={handleSetupSubmit}
          />
        )}
        {setupStatusState.phase === "loaded" && currentRoute === "/gate" && (
          <GatePage
            onMoveToChildUnlock={() => moveTo("/child/unlock")}
            onMoveToParentUnlock={() => moveTo("/parent/unlock")}
          />
        )}
        {setupStatusState.phase === "loaded" && currentRoute === "/child" && (
          <ChildPage onMoveToParentUnlock={() => moveTo("/parent/unlock")} />
        )}
        {setupStatusState.phase === "loaded" && currentRoute === "/child/unlock" && (
          <UnlockPage
            pin={childUnlockPin}
            pinLabel="아이용 PIN"
            role="child"
            unlockState={getUnlockState("child")}
            onPinChange={setChildUnlockPin}
            onSubmitUnlock={() => void handleRoleUnlockSubmit("child")}
            onMoveToGate={() => moveTo("/gate")}
            onMoveToRoleSurface={() => moveTo("/child")}
          />
        )}
        {setupStatusState.phase === "loaded" &&
          currentRoute === "/parent/unlock" && (
          <UnlockPage
            pin={parentUnlockPin}
            pinLabel="부모용 PIN"
            role="parent"
            unlockState={getUnlockState("parent")}
            onPinChange={setParentUnlockPin}
            onSubmitUnlock={() => void handleRoleUnlockSubmit("parent")}
            onMoveToGate={() => moveTo("/gate")}
            onMoveToRoleSurface={() => moveTo("/parent")}
          />
          )}
        {setupStatusState.phase === "loaded" && currentRoute === "/parent" && (
          <ParentPage
            activeSessionExpiresAt={activeSession?.sessionExpiresAt ?? null}
            onMoveToChildUnlock={() => moveTo("/child/unlock")}
            onMoveToGate={() => moveTo("/gate")}
          />
        )}
      </main>
    </div>
  );
}
