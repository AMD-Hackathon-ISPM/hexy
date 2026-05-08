import { useEffect, useMemo, useState } from 'react'
import { useViewportStore } from '@/stores/useViewportStore'

const DEFAULT_BASE_URL = ''
const PREVIEW_WIDTH = 320
const PREVIEW_HEIGHT = 180
const REFRESH_MS = 500

export function DinoDebugOverlay() {
  const dinoCamera = useViewportStore((s) => s.dinoCamera)
  const [visible, setVisible] = useState(false)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    if (!visible) return
    const timer = window.setInterval(() => {
      setTick((value) => value + 1)
    }, REFRESH_MS)
    return () => window.clearInterval(timer)
  }, [visible])

  const imageUrl = useMemo(() => {
    if (!visible) return ''
    const baseUrl = import.meta.env.VITE_HEXY_BE_URL ?? DEFAULT_BASE_URL
    const normalizedBase = baseUrl.endsWith('/')
      ? baseUrl.slice(0, -1)
      : baseUrl
    const params = new URLSearchParams({
      width: String(PREVIEW_WIDTH),
      height: String(PREVIEW_HEIGHT),
      ts: String(tick),
    })
    if (dinoCamera) {
      params.set('camera', dinoCamera)
    }
    if (!normalizedBase) {
      return `/mujoco/detections/frame?${params.toString()}`
    }
    return `${normalizedBase}/mujoco/detections/frame?${params.toString()}`
  }, [dinoCamera, tick, visible])

  return (
    <div className="hexy-dino-debug" data-visible={visible}>
      <button
        type="button"
        className="hexy-dino-debug-toggle"
        onClick={() => setVisible((prev) => !prev)}
      >
        DINO FEED
      </button>
      {visible && (
        <div className="hexy-dino-debug-panel">
          <div className="hexy-dino-debug-header">
            <span>Camera: {dinoCamera || 'default'}</span>
            <span>{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}</span>
          </div>
          <img
            className="hexy-dino-debug-frame"
            src={imageUrl}
            alt="DINO camera feed"
          />
        </div>
      )}
    </div>
  )
}

export default DinoDebugOverlay
