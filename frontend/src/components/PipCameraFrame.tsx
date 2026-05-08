import { forwardRef, type ReactNode } from 'react'
import { useViewportStore } from '@/stores/useViewportStore'
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
      <div className="hexy-pip-header">
        <span className="hexy-pip-heading">ROBOT VIEW</span>
        <div className="hexy-pip-actions">
          <button
            type="button"
            className="hexy-pip-control"
            aria-label="Switch view"
            title="Switch view"
            disabled={isTransitioning}
            onClick={swap}
          >
            <ArrowRightLeftIcon />
          </button>
          <button
            type="button"
            className="hexy-pip-control"
            aria-label="Hide PiP view"
            title="Hide PiP view"
            disabled={isTransitioning}
            onClick={() => setPipCollapsed(true)}
          >
            <Minimize2Icon />
          </button>
        </div>
      </div>
      <div ref={ref} className="hexy-pip-viewport">
        {children}
        <CameraTransitionOverlay variant="pip" />
        <div className="hexy-pip-label">ROBOT VIEW</div>
      </div>
    </div>
  )
})

export default PipCameraFrame
