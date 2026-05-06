import { useViewportStore, type TransitionPhase } from '@/stores/useViewportStore'

type CameraTransitionOverlayProps = {
  variant: 'main' | 'pip'
}

const phaseLabels: Partial<Record<TransitionPhase, string>> = {
  connecting: 'CONNECTING TO CAMERA',
  initializing: 'INITIALIZING VIEW',
}

export function CameraTransitionOverlay({ variant }: CameraTransitionOverlayProps) {
  const transitionPhase = useViewportStore((s) => s.transitionPhase)

  if (transitionPhase === 'idle') return null

  const label =
    variant === 'main' ? phaseLabels[transitionPhase] : 'SWITCHING CAMERA'

  return (
    <div
      className="hexy-camera-transition"
      data-phase={transitionPhase}
      data-variant={variant}
      aria-hidden="true"
    >
      <div className="hexy-camera-transition-surface" />
      {label && <div className="hexy-camera-transition-text">{label}</div>}
    </div>
  )
}

export default CameraTransitionOverlay
