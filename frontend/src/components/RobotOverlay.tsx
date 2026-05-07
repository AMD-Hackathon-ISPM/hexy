import { useRobotStatusStore } from '@/stores/useRobotStatusStore'

type RobotOverlayProps = {
  variant: 'main' | 'pip'
  visible?: boolean
}

export function RobotOverlay({ variant, visible = true }: RobotOverlayProps) {
  const audioAlert = useRobotStatusStore((s) => s.audioAlert)
  const audioTranscript = useRobotStatusStore((s) => s.audioTranscript)
  const dinoDetections = useRobotStatusStore((s) => s.dinoDetections)

  if (!visible) return null

  return (
    <div className={`hexy-robot-overlay hexy-robot-overlay--${variant}`}>
      {dinoDetections && dinoDetections.detections.length > 0 && (
        <div className="hexy-dino-overlay">
          {dinoDetections.detections.map((det, index) => {
            const [x1, y1, x2, y2] = det.bbox
            const width = Math.max(0, x2 - x1)
            const height = Math.max(0, y2 - y1)
            const leftPct = (x1 / dinoDetections.frameWidth) * 100
            const topPct = (y1 / dinoDetections.frameHeight) * 100
            const widthPct = (width / dinoDetections.frameWidth) * 100
            const heightPct = (height / dinoDetections.frameHeight) * 100
            return (
              <div
                key={`${det.label}-${index}`}
                className="hexy-dino-box"
                style={{
                  left: `${leftPct}%`,
                  top: `${topPct}%`,
                  width: `${widthPct}%`,
                  height: `${heightPct}%`,
                }}
              >
                <span className="hexy-dino-label">
                  {det.label} {Math.round(det.confidence * 100)}%
                </span>
              </div>
            )
          })}
        </div>
      )}
      {audioAlert && (
        <div className="hexy-audio-alert">
          <span className="hexy-audio-alert-label">AUDIO ALERT</span>
          <span className="hexy-audio-alert-keyword">{audioAlert.keyword}</span>
          {audioTranscript?.direction && (
            <span className="hexy-audio-alert-direction">
              {audioTranscript.direction.toUpperCase()}
            </span>
          )}
          {audioTranscript?.distanceM !== undefined && (
            <span className="hexy-audio-alert-distance">
              {Math.round(audioTranscript.distanceM)}m
            </span>
          )}
        </div>
      )}
      {audioTranscript?.text && (
        <div className="hexy-audio-transcript">
          “{audioTranscript.text}”
        </div>
      )}
    </div>
  )
}

export default RobotOverlay
