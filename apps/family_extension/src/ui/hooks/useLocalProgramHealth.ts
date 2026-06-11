import { useEffect, useState } from "react";

import { checkLocalProgramHealth } from "../api/wellbeing";

export type LocalProgramHealthState = "checking" | "connected" | "offline";

export function useLocalProgramHealth() {
  const [healthState, setHealthState] =
    useState<LocalProgramHealthState>("checking");

  useEffect(() => {
    let cancelled = false;

    async function loadHealth() {
      const isConnected = await checkLocalProgramHealth();
      if (!cancelled) {
        setHealthState(isConnected ? "connected" : "offline");
      }
    }

    void loadHealth();
    const intervalId = window.setInterval(() => {
      void loadHealth();
    }, 15_000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  return healthState;
}
