import { useEffect, useRef, useState } from "react";
import { HexapodLoader } from "./HexapodLoader";

type MujocoLoadingOverlayProps = {
  exiting?: boolean;
  ready?: boolean;
  error?: string | null;
  detailOverride?: string | null;
};

const STAGES = [
  "Detecting scene",
  "Fetching assets",
  "Compiling physics",
  "Warming actuators",
] as const;

const STAGE_INTERVAL_MS = 1400;
const STUCK_HINT_AFTER_S = 15;

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs.toString().padStart(2, "0")}s`;
}

export function MujocoLoadingOverlay({
  exiting = false,
  ready = false,
  error = null,
  detailOverride = null,
}: MujocoLoadingOverlayProps) {
  const [stageIndex, setStageIndex] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const startedAt = useRef<number>(performance.now());

  useEffect(() => {
    if (ready || error) return;
    const stageTimer = window.setInterval(() => {
      setStageIndex((i) => (i + 1) % STAGES.length);
    }, STAGE_INTERVAL_MS);
    const tickTimer = window.setInterval(() => {
      setElapsed(Math.floor((performance.now() - startedAt.current) / 1000));
    }, 1000);
    return () => {
      window.clearInterval(stageTimer);
      window.clearInterval(tickTimer);
    };
  }, [ready, error]);

  const label = error
    ? `Scene load failed: ${error}`
    : ready
      ? "Initializing Simulation"
      : "Preparing Hexy";

  let detail: string;
  if (error) {
    detail = "Check the backend logs and reload";
  } else if (ready) {
    detail = "Linking sensors";
  } else if (detailOverride) {
    detail = `${detailOverride} — ${formatElapsed(elapsed)}`;
  } else {
    detail = `${STAGES[stageIndex]} — ${formatElapsed(elapsed)}`;
  }

  const showStuckHint =
    !ready && !error && elapsed >= STUCK_HINT_AFTER_S;

  return (
    <HexapodLoader
      exiting={exiting}
      label={label}
      detail={detail}
      hint={
        showStuckHint
          ? "Taking longer than usual. Open DevTools (F12) → Console for errors."
          : null
      }
      showProgress={!error}
      ready={ready}
    />
  );
}

export default MujocoLoadingOverlay;
