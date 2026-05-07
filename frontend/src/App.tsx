import { lazy, Suspense, useEffect, useState } from 'react'
import { AssistantChatRuntimeProvider } from './AssistantChatRuntimeProvider'
import { FloatingAssistantChat } from './FloatingAssistantChat'
import { HeaderLabels } from './components/HeaderLabels'
import { FloatingInfoPanel } from './components/FloatingInfoPanel'
import { MujocoLoadingOverlay } from './components/MujocoLoadingOverlay'
import { useMujocoStream } from './hooks/useMujocoStream'

const RobotScene = lazy(() => import('./components/RobotScene'))
const LOADER_READY_PAUSE_MS = 5000
const LOADER_FADE_MS = 320

function App() {
  useMujocoStream()
  const [hexyReady, setHexyReady] = useState(false)
  const [hexyError, setHexyError] = useState<string | null>(null)
  const [hexyLoadDetail, setHexyLoadDetail] = useState<string | null>(null)
  const [showLoadingOverlay, setShowLoadingOverlay] = useState(true)
  const [loadingOverlayExiting, setLoadingOverlayExiting] = useState(false)

  useEffect(() => {
    if (!hexyReady || hexyError) return

    const fadeTimer = window.setTimeout(() => {
      setLoadingOverlayExiting(true)
    }, LOADER_READY_PAUSE_MS)
    const removeTimer = window.setTimeout(() => {
      setShowLoadingOverlay(false)
    }, LOADER_READY_PAUSE_MS + LOADER_FADE_MS)

    return () => {
      window.clearTimeout(fadeTimer)
      window.clearTimeout(removeTimer)
    }
  }, [hexyError, hexyReady])

  const showHud = !showLoadingOverlay

  return (
    <AssistantChatRuntimeProvider>
      <Suspense
        fallback={
          <div className="hexy-scene" />
        }
      >
        <RobotScene
          onLoadError={setHexyError}
          onLoadDetail={setHexyLoadDetail}
          onMainReady={() => {
            setHexyError(null)
            setHexyReady(true)
          }}
          showPip={showHud}
        />
      </Suspense>
      {showHud && (
        <>
          <HeaderLabels />
          <FloatingInfoPanel />
          <FloatingAssistantChat />
        </>
      )}
      {showLoadingOverlay && (
        <MujocoLoadingOverlay
          error={hexyError}
          detailOverride={hexyLoadDetail}
          ready={hexyReady}
          exiting={loadingOverlayExiting}
        />
      )}
    </AssistantChatRuntimeProvider>
  )
}

export default App
