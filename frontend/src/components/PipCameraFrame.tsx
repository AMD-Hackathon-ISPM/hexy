import { forwardRef, type ReactNode } from 'react'
import { useViewportStore } from '@/stores/useViewportStore'
import { useRobotStatusStore } from '@/stores/useRobotStatusStore'
import {
  ArrowRightLeftIcon,
  Minimize2Icon,
  PictureInPicture2Icon,
} from 'lucide-react'
import { CameraTransitionOverlay } from './CameraTransitionOverlay'

type PipCameraFrameProps = {
  children?: ReactNode
  collapsed?: boolean
  hasCanvas?: boolean
}

export const PipCameraFrame = forwardRef<HTMLDivElement, PipCameraFrameProps>(
function PipCameraFrame({ children, collapsed = false, hasCanvas: hasCanvasProp }, ref) {
  const swap = useViewportStore((s) => s.swap)
  const setPipCollapsed = useViewportStore((s) => s.setPipCollapsed)
  const transitionPhase = useViewportStore((s) => s.transitionPhase)
  const audioAlert = useRobotStatusStore((s) => s.audioAlert)
  const audioTranscript = useRobotStatusStore((s) => s.audioTranscript)
  const hasCanvas = hasCanvasProp ?? Boolean(children)
  const isTransitioning = transitionPhase !== 'idle'

  if (collapsed) {
    return (
      <button
        type="button"
        className="hexy-pip-restore"
        aria-label="Show PiP view"
        onClick={() => setPipCollapsed(false)}
      >
        <PictureInPicture2Icon />
      </button>
    )
  }

  return (
    <div
      className="hexy-pip"
      data-has-canvas={hasCanvas}
      data-transitioning={isTransitioning}
    >
      <div ref={ref} className="hexy-pip-viewport">
        {children}
        <CameraTransitionOverlay variant="pip" />
      </div>
      {audioAlert && (
        <div className="hexy-audio-alert">
          <span className="hexy-audio-alert-label">AUDIO ALERT</span>
          <span className="hexy-audio-alert-keyword">{audioAlert.keyword}</span>
          {audioTranscript?.direction && (
            <span className="hexy-audio-alert-direction">
              {audioTranscript.direction.toUpperCase()}
            </span>
          )}
        </div>
      )}
      {audioTranscript?.text && (
        <div className="hexy-audio-transcript">
          “{audioTranscript.text}”
        </div>
      )}
      <button
        type="button"
        className="hexy-pip-expand"
        aria-label="Switch view"
        disabled={isTransitioning}
        onClick={swap}
      >
        <ArrowRightLeftIcon />
      </button>
      <button
        type="button"
        className="hexy-pip-expand hexy-pip-collapse"
        aria-label="Hide PiP view"
        disabled={isTransitioning}
        onClick={() => setPipCollapsed(true)}
      >
        <Minimize2Icon />
      </button>
      <div className="hexy-pip-label">ROBOT VIEW</div>
    </div>
  )
})

export default PipCameraFrame
