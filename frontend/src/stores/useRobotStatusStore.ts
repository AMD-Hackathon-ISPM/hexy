import { create } from 'zustand'
import type { MujocoState } from '@/lib/backendClient'

export type TaskStatus = 'idle' | 'active' | 'done'

export type Task = {
  id: string
  name: string
  status: TaskStatus
  progress?: number
}

export type ReasoningStepStatus = 'pending' | 'active' | 'done'

export type ReasoningStep = {
  id: string
  text: string
  status: ReasoningStepStatus
  timestamp: number
}

export type RobotStatus = {
  battery?: number
  connection?: 'online' | 'offline'
  mode?: string
  simTime?: number
  contacts?: number
  agentMode?: string
  agentAction?: string
}

export type AudioTranscript = {
  text: string
  timestamp: number
  direction?: string
  pan?: number
  distanceM?: number
  rms?: number
  sourceId?: string
}

export type AudioAlert = {
  keyword: string
  timestamp: number
}

export type SurvivorAudioEvent = {
  id: number
  phrase: string
  sourceId: string
  timestamp: number
}

type SurvivorAudioEventInput = Omit<SurvivorAudioEvent, 'id'>

export type DinoDetection = {
  label: string
  confidence: number
  bbox: [number, number, number, number]
}

export type DinoDetections = {
  detections: DinoDetection[]
  frameWidth: number
  frameHeight: number
  updatedAt: number
}

type RobotStatusState = {
  tasks: Task[]
  status: RobotStatus
  reasoningSteps: ReasoningStep[]
  mujocoState?: MujocoState
  audioTranscript?: AudioTranscript
  audioAlert?: AudioAlert
  survivorAudioEvent?: SurvivorAudioEvent
  dinoDetections?: DinoDetections
  setTasks: (tasks: Task[]) => void
  setStatus: (patch: Partial<RobotStatus>) => void
  setReasoningSteps: (steps: ReasoningStep[]) => void
  addReasoningStep: (step: Omit<ReasoningStep, 'id'>) => void
  setMujocoState: (state: MujocoState) => void
  setMujocoStreamState: (state: MujocoState) => void
  setAudioTranscript: (transcript: AudioTranscript) => void
  setAudioAlert: (alert?: AudioAlert) => void
  setSurvivorAudioEvent: (event: SurvivorAudioEventInput) => void
  setDinoDetections: (detections: DinoDetections) => void
}

const initialTasks: Task[] = [
  { id: 'search', name: 'Cave search-and-rescue', status: 'active', progress: 0 },
]

const initialStatus: RobotStatus = {
  connection: 'online',
  mode: 'MuJoCo',
  simTime: 0,
  contacts: 0,
  agentMode: 'Initialising',
  agentAction: '—',
}

let _reasoningCounter = 0

export const useRobotStatusStore = create<RobotStatusState>((set) => ({
  tasks: initialTasks,
  status: initialStatus,
  reasoningSteps: [],
  mujocoState: undefined,
  audioTranscript: undefined,
  audioAlert: undefined,
  survivorAudioEvent: undefined,
  dinoDetections: undefined,
  setTasks: (tasks) => set({ tasks }),
  setStatus: (patch) =>
    set((state) => ({ status: { ...state.status, ...patch } })),
  setReasoningSteps: (steps) => set({ reasoningSteps: steps }),
  addReasoningStep: (step) =>
    set((state) => ({
      reasoningSteps: [
        { ...step, id: `r${++_reasoningCounter}` },
        ...state.reasoningSteps.slice(0, 49), // keep last 50
      ],
    })),
  setMujocoState: (state) => set({ mujocoState: state }),
  setMujocoStreamState: (mujocoState) =>
    set((state) => ({
      mujocoState,
      status: {
        ...state.status,
        simTime: mujocoState.time,
        contacts: mujocoState.ncon,
        connection: 'online',
        mode: 'MuJoCo Stream',
      },
    })),
  setAudioTranscript: (audioTranscript) => set({ audioTranscript }),
  setAudioAlert: (audioAlert) => set({ audioAlert }),
  setSurvivorAudioEvent: (event) =>
    set((state) => ({
      survivorAudioEvent: {
        ...event,
        id: (state.survivorAudioEvent?.id ?? 0) + 1,
      },
    })),
  setDinoDetections: (dinoDetections) => set({ dinoDetections }),
}))
