import { useViewportStore } from '@/stores/useViewportStore'

export function HeaderLabels() {
  const pipSlot = useViewportStore((s) => s.pipSlot)
  const mainViewLabel = pipSlot === 'robotPOV' ? 'Environment View' : 'Robot View'

  return (
    <>
      <div className="hexy-header hexy-header-left">HEXY</div>
      <div className="hexy-header hexy-header-right">{mainViewLabel}</div>
    </>
  )
}

export default HeaderLabels
