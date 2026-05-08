import { lazy, Suspense, useEffect, useState } from 'react'
import { stepMujoco } from './lib/backendClient'
import { useRobotStatusStore } from './stores/useRobotStatusStore'
import { AssistantChatRuntimeProvider } from './AssistantChatRuntimeProvider'
import { FloatingAssistantChat } from './FloatingAssistantChat'
import { HeaderLabels } from './components/HeaderLabels'
import { FloatingInfoPanel } from './components/FloatingInfoPanel'
import { MujocoLoadingOverlay } from './components/MujocoLoadingOverlay'
import { DinoDebugOverlay } from './components/DinoDebugOverlay'
import { useMujocoStream } from './hooks/useMujocoStream'
import { useWhisperStream } from './hooks/useWhisperStream'
import { useDinoDetections } from './hooks/useDinoDetections'

const RobotScene = lazy(() => import('./components/RobotScene'))
const LOADER_READY_PAUSE_MS = 5000
const LOADER_FADE_MS = 320
const MOVEMENT_STEP_MS = 50
const MOVEMENT_KEYS = ['w', 'a', 's', 'd'] as const

type MovementKey = (typeof MOVEMENT_KEYS)[number]

function isMovementKey(key: string): key is MovementKey {
  return MOVEMENT_KEYS.includes(key as MovementKey)
}

function App() {
  useMujocoStream()
  useWhisperStream()
  useDinoDetections()
  const [hexyReady, setHexyReady] = useState(false)
  const [hexyError, setHexyError] = useState<string | null>(null)
  const [hexyLoadDetail, setHexyLoadDetail] = useState<string | null>(null)
  const [showLoadingOverlay, setShowLoadingOverlay] = useState(true)
  const [loadingOverlayExiting, setLoadingOverlayExiting] = useState(false)

  useEffect(() => {
    const pressedKeys = new Set<MovementKey>()
    let activeKey: MovementKey | null = null
    let intervalId: number | null = null
    let inFlight = false

    const sendMovementStep = () => {
      if (!activeKey || inFlight) return

      inFlight = true
      stepMujoco({ key: activeKey, n_steps: 4 })
        .then((state) => {
          useRobotStatusStore.getState().setMujocoStreamState(state)
        })
        .catch(() => {
          // ignore transient movement errors while the backend reconnects
        })
        .finally(() => {
          inFlight = false
        })
    }

    const startMovementLoop = () => {
      if (intervalId !== null) return
      sendMovementStep()
      intervalId = window.setInterval(sendMovementStep, MOVEMENT_STEP_MS)
    }

    const stopMovementLoop = () => {
      if (intervalId !== null) {
        window.clearInterval(intervalId)
        intervalId = null
      }
      stepMujoco({ n_steps: 1 }).catch(() => {
        // ignore movement errors during shutdown
      })
    }

    const isEditableTarget = (target: EventTarget | null) => (
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      (target instanceof HTMLElement && target.isContentEditable)
    )

    const handleKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase()
      if (!isMovementKey(key)) return
      if (event.altKey || event.ctrlKey || event.metaKey) return
      if (isEditableTarget(event.target)) return

      event.preventDefault()
      pressedKeys.add(key)
      activeKey = key
      startMovementLoop()
    }

    const handleKeyUp = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase()
      if (!isMovementKey(key)) return

      pressedKeys.delete(key)
      const remainingKeys = Array.from(pressedKeys)
      activeKey = remainingKeys.length > 0 ? remainingKeys[remainingKeys.length - 1] : null
      if (!activeKey) stopMovementLoop()
    }

    const handleBlur = () => {
      pressedKeys.clear()
      activeKey = null
      stopMovementLoop()
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    window.addEventListener('blur', handleBlur)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('blur', handleBlur)
      if (intervalId !== null) window.clearInterval(intervalId)
    }
  }, [])

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
          <DinoDebugOverlay />
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
