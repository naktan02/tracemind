import { useEffect, useMemo, useState } from "react";

import { ConnectionStateBanner } from "./components/ConnectionStateBanner";
import counselingHeroUrl from "./assets/counseling-hero.png";
import { useFamilyAccess } from "./hooks/useFamilyAccess";
import { useLocalProgramHealth } from "./hooks/useLocalProgramHealth";
import { ChildPage, type ChildTab } from "./pages/child/ChildPage";
import { ParentPage } from "./pages/parent/ParentPage";
import { SetupPage } from "./pages/setup/SetupPage";
import { UnlockPage } from "./pages/unlock/UnlockPage";
import {
  AppRoute,
  ROUTE_META,
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
  const fallbackRoute = normalizeRoute(initialRoute, "/child/unlock");
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
  const [childActiveTab, setChildActiveTab] = useState<ChildTab>("analysis");
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

  function moveToSelfHome() {
    setChildActiveTab("analysis");
    if (activeRole !== "child") {
      return;
    }
    setCurrentRoute("/child");
    updateHash("/child");
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
      setCurrentRoute("/child/unlock");
      updateHash("/child/unlock");
    }
  }

  return (
    <div className="app-shell">
      <header className="top-bar">
        <button
          className="brand-mark"
          type="button"
          onClick={moveToSelfHome}
        >
          <span className="brand-name">TraceMind</span>
        </button>
        {currentRoute === "/child" && activeRole === "child" && (
          <nav className="top-nav" aria-label="본인 페이지 탭">
            {(
              [
                ["analysis", "그래프 및 분석"],
                ["ai", "AI 마음 도움"],
                ["checkin", "자기 점검"],
              ] as const
            ).map(([tab, label]) => (
              <button
                key={tab}
                className={childActiveTab === tab ? "active" : undefined}
                type="button"
                onClick={() => setChildActiveTab(tab)}
              >
                {label}
              </button>
            ))}
          </nav>
        )}
        {!isSetupComplete && (
          <nav className="top-nav" aria-label="처음 설정">
            <button type="button" onClick={() => moveTo("/setup")}>
              처음 설정
            </button>
          </nav>
        )}
        <div className="top-spacer" aria-hidden="true" />
      </header>

      <main className="main-panel">
        <header
          className="page-hero"
          style={{ backgroundImage: `url(${counselingHeroUrl})` }}
        >
          <div className="page-hero-content">
            <p className="page-eyebrow">{routeMeta.eyebrow}</p>
            <h2>{routeMeta.title}</h2>
            <p className="page-description">{routeMeta.description}</p>
          </div>
        </header>
        <ConnectionStateBanner healthState={healthState} />

        {setupStatusState.phase === "loading" && (
          <section className="hero-card gate-hero">
            <div>
              <p className="eyebrow">설정 확인</p>
              <h2>초기 설정 상태를 확인하는 중</h2>
              <p className="section-copy">
                이 기기에서 사용할 본인/부모 화면 설정을 확인하고 있습니다.
              </p>
            </div>
            <div className="hero-meter">
              <span className="hero-meter-label">설정 상태</span>
              <strong>확인 중...</strong>
            </div>
          </section>
        )}
        {setupStatusState.phase === "error" && (
          <section className="hero-card gate-hero">
            <div>
              <p className="eyebrow">설정 확인</p>
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
        {setupStatusState.phase === "loaded" && currentRoute === "/child" && (
          <ChildPage activeTab={childActiveTab} />
        )}
        {setupStatusState.phase === "loaded" && currentRoute === "/child/unlock" && (
          <UnlockPage
            pin={childUnlockPin}
            pinLabel="본인 PIN"
            role="child"
            unlockState={getUnlockState("child")}
            onPinChange={setChildUnlockPin}
            onSubmitUnlock={() => void handleRoleUnlockSubmit("child")}
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
            onMoveToRoleSurface={() => moveTo("/parent")}
          />
          )}
        {setupStatusState.phase === "loaded" && currentRoute === "/parent" && (
          <ParentPage
            activeSessionExpiresAt={activeSession?.sessionExpiresAt ?? null}
          />
        )}
      </main>
    </div>
  );
}
