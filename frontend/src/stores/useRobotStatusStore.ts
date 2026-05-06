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
}

type RobotStatusState = {
  tasks: Task[]
  status: RobotStatus
  reasoningSteps: ReasoningStep[]
  mujocoState?: MujocoState
  setTasks: (tasks: Task[]) => void
  setStatus: (patch: Partial<RobotStatus>) => void
  setReasoningSteps: (steps: ReasoningStep[]) => void
  setMujocoState: (state: MujocoState) => void
  setMujocoStreamState: (state: MujocoState) => void
}

const demoTasks: Task[] = [
  {
    id: 'scan-workcell',
    name: 'Scan workcell',
    status: 'done',
    progress: 1,
  },
  {
    id: 'align-gripper',
    name: 'Align gripper',
    status: 'active',
    progress: 0.64,
  },
  {
    id: 'place-cube',
    name: 'Place cube',
    status: 'idle',
    progress: 0,
  },
]

const demoStatus: RobotStatus = {
  battery: 87,
  connection: 'online',
  mode: 'Assisted',
  simTime: 0,
  contacts: 0,
}

const demoReasoningSteps: ReasoningStep[] = [
  { id: 'r1', text: 'Reason 1', status: 'done', timestamp: Date.now() - 60000 },
  { id: 'r2', text: 'Reason 2', status: 'done', timestamp: Date.now() - 30000 },
  { id: 'r3', text: 'Reason 3', status: 'active', timestamp: Date.now() - 10000 },
  { id: 'r4', text: 'Reason 4', status: 'pending', timestamp: 0 },
]

export const useRobotStatusStore = create<RobotStatusState>((set) => ({
  tasks: demoTasks,
  status: demoStatus,
  reasoningSteps: demoReasoningSteps,
  mujocoState: undefined,
  setTasks: (tasks) => set({ tasks }),
  setStatus: (patch) =>
    set((state) => ({ status: { ...state.status, ...patch } })),
  setReasoningSteps: (steps) => set({ reasoningSteps: steps }),
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
}))
