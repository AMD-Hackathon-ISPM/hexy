import { RotateCcw } from 'lucide-react'
import { useViewportStore, type ViewMode } from '@/stores/useViewportStore'

const VIEW_MODE_OPTIONS: { value: ViewMode; label: string }[] = [
  { value: 'orbital', label: 'Orbital' },
  { value: 'freecam', label: 'Freecam' },
]

const DINO_CAMERA_OPTIONS = [
  { value: '', label: 'Default' },
  { value: 'robot_pov', label: 'Robot POV' },
  { value: 'robot_cam', label: 'Robot Cam' },
  { value: 'dataset_cam', label: 'Dataset Cam' },
]

export function HeaderLabels() {
  const pipSlot = useViewportStore((s) => s.pipSlot)
  const viewMode = useViewportStore((s) => s.viewMode)
  const setViewMode = useViewportStore((s) => s.setViewMode)
  const resetFreecamCamera = useViewportStore((s) => s.resetFreecamCamera)
  const dinoCamera = useViewportStore((s) => s.dinoCamera)
  const setDinoCamera = useViewportStore((s) => s.setDinoCamera)
  const mainViewLabel =
    viewMode === 'freecam'
      ? 'Free Camera'
      : pipSlot === 'robotPOV'
        ? 'Environment View'
        : 'Robot View'

  return (
    <>
      <div className="hexy-header hexy-header-left">HEXY</div>
      <div className="hexy-header hexy-header-right">
        <div className="hexy-header-label">{mainViewLabel}</div>
        {pipSlot === 'robotPOV' && (
          <div className="hexy-camera-controls">
            <label className="hexy-camera-picker">
              <span className="hexy-camera-picker-label">DINO</span>
              <select
                className="hexy-camera-select"
                value={dinoCamera}
                onChange={(event) => setDinoCamera(event.target.value)}
              >
                {DINO_CAMERA_OPTIONS.map((option) => (
                  <option key={option.value || 'default'} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="hexy-view-mode-toggle" role="group" aria-label="Camera mode">
              {VIEW_MODE_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  className="hexy-view-mode-option"
                  data-active={viewMode === option.value}
                  onClick={() => setViewMode(option.value)}
                >
                  {option.label}
                </button>
              ))}
            </div>
            {viewMode === 'freecam' && (
              <button
                type="button"
                className="hexy-camera-reset"
                aria-label="Reset free camera"
                title="Reset free camera"
                onClick={resetFreecamCamera}
              >
                <RotateCcw size={14} strokeWidth={2.4} />
              </button>
            )}
          </div>
        )}
      </div>
    </>
  )
}

export default HeaderLabels
