import { forwardRef, useEffect, useRef, useState, type ReactNode } from 'react'
import { useViewportStore } from '@/stores/useViewportStore'
import {
  ArrowRightFromLineIcon,
  ArrowRightLeftIcon,
  PictureInPicture2Icon,
} from 'lucide-react'
import { CameraTransitionOverlay } from './CameraTransitionOverlay'

type PipCameraFrameProps = {
  children?: ReactNode
  collapsed?: boolean
  hasCanvas?: boolean
}

const PIP_EXIT_ANIMATION_MS = 300

export const PipCameraFrame = forwardRef<HTMLDivElement, PipCameraFrameProps>(
function PipCameraFrame({ children, collapsed = false, hasCanvas: hasCanvasProp }, ref) {
  const swap = useViewportStore((s) => s.swap)
  const setPipCollapsed = useViewportStore((s) => s.setPipCollapsed)
  const transitionPhase = useViewportStore((s) => s.transitionPhase)
  const viewMode = useViewportStore((s) => s.viewMode)
  const [closing, setClosing] = useState(false)
  const closeTimerRef = useRef<number | null>(null)
  const hasCanvas = hasCanvasProp ?? Boolean(children)
  const isTransitioning = transitionPhase !== 'idle'
  const switchDisabled = isTransitioning || viewMode === 'freecam' || closing

  useEffect(() => {
    if (!collapsed) setClosing(false)
  }, [collapsed])

  useEffect(() => () => {
    if (closeTimerRef.current !== null) {
      window.clearTimeout(closeTimerRef.current)
    }
  }, [])

  const closePip = () => {
    if (isTransitioning || closing) return

    setClosing(true)
    closeTimerRef.current = window.setTimeout(() => {
      closeTimerRef.current = null
      setPipCollapsed(true)
    }, PIP_EXIT_ANIMATION_MS)
  }

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
      data-closing={closing}
    >
      <div className="hexy-pip-header">
        <span className="hexy-pip-heading">ROBOT VIEW</span>
        <div className="hexy-pip-actions">
          <button
            type="button"
            className="hexy-pip-control"
            aria-label="Switch view"
            title="Switch view"
            disabled={switchDisabled}
            onClick={swap}
          >
            <ArrowRightLeftIcon />
          </button>
          <button
            type="button"
            className="hexy-pip-control"
            aria-label="Hide PiP view"
            title="Hide PiP view"
            disabled={isTransitioning || closing}
            onClick={closePip}
          >
            <ArrowRightFromLineIcon />
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
