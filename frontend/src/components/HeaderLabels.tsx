import { useViewportStore, type ViewMode } from '@/stores/useViewportStore'

const VIEW_MODE_OPTIONS: { value: ViewMode; label: string }[] = [
  { value: 'orbital', label: 'Orbital' },
  { value: 'freecam', label: 'Freecam' },
]

export function HeaderLabels() {
  const pipSlot = useViewportStore((s) => s.pipSlot)
  const viewMode = useViewportStore((s) => s.viewMode)
  const setViewMode = useViewportStore((s) => s.setViewMode)
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
        )}
      </div>
    </>
  )
}

export default HeaderLabels
