import { create } from 'zustand'

export type CameraPreset = 'orbit' | 'robotPOV'
export type ViewMode = 'orbital' | 'freecam'
export type TransitionPhase =
  | 'idle'
  | 'static'
  | 'connecting'
  | 'initializing'
  | 'revealing'

const DEFAULT_DINO_CAMERA = import.meta.env.VITE_HEXY_DINO_CAMERA ?? 'robot_pov'

type ViewportState = {
  pipSlot: CameraPreset
  pipCollapsed: boolean
  transitionPhase: TransitionPhase
  viewMode: ViewMode
  freecamResetNonce: number
  dinoCamera: string
  setPipCollapsed: (collapsed: boolean) => void
  swap: () => void
  setViewMode: (mode: ViewMode) => void
  resetFreecamCamera: () => void
  setDinoCamera: (camera: string) => void
}

let transitionTimers: number[] = []

function clearTransitionTimers() {
  transitionTimers.forEach((timer) => window.clearTimeout(timer))
  transitionTimers = []
}

function queueTransitionTimer(callback: () => void, delay: number) {
  transitionTimers.push(window.setTimeout(callback, delay))
}

function getSwappedSlot(slot: CameraPreset): CameraPreset {
  return slot === 'robotPOV' ? 'orbit' : 'robotPOV'
}

export const useViewportStore = create<ViewportState>((set, get) => ({
  pipSlot: 'robotPOV',
  pipCollapsed: false,
  transitionPhase: 'idle',
  viewMode: 'orbital',
  freecamResetNonce: 0,
  dinoCamera: DEFAULT_DINO_CAMERA,
  setPipCollapsed: (pipCollapsed) => set({ pipCollapsed }),
  setViewMode: (viewMode) =>
    set((state) => ({
      viewMode,
      pipSlot: viewMode === 'freecam' ? 'robotPOV' : state.pipSlot,
    })),
  resetFreecamCamera: () =>
    set((state) => ({ freecamResetNonce: state.freecamResetNonce + 1 })),
  setDinoCamera: (dinoCamera) => set({ dinoCamera }),
  swap: () => {
    if (get().viewMode === 'freecam') return
    if (get().transitionPhase !== 'idle') return

    clearTransitionTimers()
    set({ transitionPhase: 'static' })

    queueTransitionTimer(() => {
      set({ transitionPhase: 'connecting' })
    }, 150)

    queueTransitionTimer(() => {
      set((state) => ({
        pipSlot: getSwappedSlot(state.pipSlot),
      }))
    }, 700)

    queueTransitionTimer(() => {
      set({ transitionPhase: 'initializing' })
    }, 750)

    queueTransitionTimer(() => {
      set({ transitionPhase: 'revealing' })
    }, 1300)

    queueTransitionTimer(() => {
      set({ transitionPhase: 'idle' })
      clearTransitionTimers()
    }, 1500)
  },
}))
