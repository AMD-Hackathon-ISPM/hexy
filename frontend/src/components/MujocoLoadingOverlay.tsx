import { HexapodLoader } from "./HexapodLoader";

type MujocoLoadingOverlayProps = {
  exiting?: boolean;
  ready?: boolean;
  error?: string | null;
};

export function MujocoLoadingOverlay({
  exiting = false,
  ready = false,
  error = null,
}: MujocoLoadingOverlayProps) {
  const label = error
    ? `Scene load failed: ${error}`
    : ready
      ? "Initializing Simulation"
      : "Preparing Hexy";

  return <HexapodLoader exiting={exiting} label={label} ready={ready} />;
}

export default MujocoLoadingOverlay;
