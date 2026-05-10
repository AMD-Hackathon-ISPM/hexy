import { useState, type ReactNode } from 'react'
import { useRobotStatusStore } from '@/stores/useRobotStatusStore'
import type { RobotStatus, Task, ReasoningStep } from '@/stores/useRobotStatusStore'
import { Collapsible, CollapsibleContent } from '@/components/ui/collapsible'
import { cn } from '@/lib/utils'
import {
  ArrowLeftToLineIcon,
  MinusIcon,
  PanelLeftOpenIcon,
  PlusIcon,
} from 'lucide-react'

const STATUS_LABELS: Record<string, string> = {
  battery: 'Battery',
  connection: 'Connection',
  mode: 'Mode',
  simTime: 'Sim Time',
  contacts: 'Contacts',
  agentMode: 'Agent',
  agentAction: 'Action',
}

function formatStatusValue(key: string, value: unknown): string {
  if (key === 'battery' && typeof value === 'number') return `${value}%`
  if (key === 'simTime' && typeof value === 'number') return `${value.toFixed(2)}s`
  return String(value ?? '')
}

function TaskRow({ task }: { task: Task }) {
  return (
    <div className="hexy-info-row">
      <span className="hexy-info-row-dot" data-status={task.status} />
      <span className="hexy-info-row-name">{task.name}</span>
      {typeof task.progress === 'number' && (
        <span className="hexy-info-row-progress">
          {Math.round(task.progress * 100)}%
        </span>
      )}
    </div>
  )
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="hexy-info-row">
      <span className="hexy-info-row-key">{label}</span>
      <span className="hexy-info-row-value">{value}</span>
    </div>
  )
}

function ReasoningStepRow({ step }: { step: ReasoningStep }) {
  return (
    <div className="hexy-info-row">
      <span className="hexy-info-row-dot" data-status={step.status} />
      <span className="hexy-info-row-name">{step.text}</span>
      <span className="hexy-info-row-progress">
        {step.timestamp > 0
          ? new Date(step.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          : '—'}
      </span>
    </div>
  )
}

function SectionToggleButton({ collapsed, onClick }: { collapsed: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      className="hexy-section-toggle"
      onClick={(e) => { e.stopPropagation(); onClick() }}
      aria-label={collapsed ? 'Expand section' : 'Collapse section'}
    >
      {collapsed ? <PlusIcon /> : <MinusIcon />}
    </button>
  )
}

type InfoSectionProps = {
  id: string
  label: string
  collapsed: boolean
  onToggle: () => void
  children: ReactNode
}

function InfoSection({ id, label, collapsed, onToggle, children }: InfoSectionProps) {
  return (
    <>
      <div className="hexy-info-section-header" data-section={id}>
        <span className="hexy-info-section-label">{label}</span>
        <SectionToggleButton collapsed={collapsed} onClick={onToggle} />
      </div>
      <Collapsible open={!collapsed}>
        <CollapsibleContent
          data-section={id}
          className={cn(
            'hexy-info-section-body',
            'overflow-hidden',
            'data-[state=closed]:animate-collapsible-up',
            'data-[state=open]:animate-collapsible-down',
          )}
        >
          {children}
        </CollapsibleContent>
      </Collapsible>
    </>
  )
}

function TabBar({ tab, onTabChange }: { tab: string; onTabChange: (tab: 'execution' | 'reasoning') => void }) {
  return (
    <div className="hexy-panel-tabs">
      <button
        type="button"
        className="hexy-panel-tab"
        data-active={tab === 'execution'}
        onClick={() => onTabChange('execution')}
      >
        Execution
      </button>
      <button
        type="button"
        className="hexy-panel-tab"
        data-active={tab === 'reasoning'}
        onClick={() => onTabChange('reasoning')}
      >
        Reasoning
      </button>
    </div>
  )
}

export function FloatingInfoPanel() {
  const [panelVisible, setPanelVisible] = useState(true)
  const [collapsedSections, setCollapsedSections] = useState<Set<string>>(new Set())
  const [tab, setTab] = useState<'execution' | 'reasoning'>('execution')

  const tasks = useRobotStatusStore((s) => s.tasks)
  const status = useRobotStatusStore((s) => s.status)
  const reasoningSteps = useRobotStatusStore((s) => s.reasoningSteps)

  const toggleSection = (sectionId: string) => {
    setCollapsedSections((prev) => {
      const next = new Set(prev)
      if (next.has(sectionId)) {
        next.delete(sectionId)
      } else {
        next.add(sectionId)
      }
      return next
    })
  }

  const isCollapsed = (sectionId: string) => collapsedSections.has(sectionId)

  const statusEntries = (Object.keys(status) as Array<keyof RobotStatus>).filter(
    (key) => status[key] !== undefined,
  )

  return (
    <>
      <div className="hexy-info-panel" data-visible={panelVisible}>
        <div className="hexy-panel-header-row">
          <TabBar tab={tab} onTabChange={setTab} />
        </div>
        <div className="hexy-panel-scroll">
          {tab === 'execution' && (
            <>
              <InfoSection
                id="task"
                label="CURRENT TASK"
                collapsed={isCollapsed('task')}
                onToggle={() => toggleSection('task')}
              >
                {tasks.map((task) => (
                  <TaskRow key={task.id} task={task} />
                ))}
              </InfoSection>
              {!isCollapsed('task') && <div className="hexy-panel-spacer" />}
              <InfoSection
                id="status"
                label="STATUS"
                collapsed={isCollapsed('status')}
                onToggle={() => toggleSection('status')}
              >
                {statusEntries.map((key) => (
                  <StatusRow
                    key={key}
                    label={STATUS_LABELS[key]}
                    value={formatStatusValue(key, status[key])}
                  />
                ))}
              </InfoSection>
            </>
          )}
          {tab === 'reasoning' && (
            <InfoSection
              id="reasoning"
              label="REASONING"
              collapsed={isCollapsed('reasoning')}
              onToggle={() => toggleSection('reasoning')}
            >
              {reasoningSteps.map((step) => (
                <ReasoningStepRow key={step.id} step={step} />
              ))}
            </InfoSection>
          )}
        </div>
        <div className="hexy-panel-footer-row">
          <button
            type="button"
            className="hexy-panel-hide-btn"
            onClick={() => setPanelVisible(false)}
            aria-label="Hide panel"
          >
            <ArrowLeftToLineIcon />
          </button>
        </div>
      </div>
      <button
        type="button"
        className="hexy-panel-expand-btn"
        data-visible={!panelVisible}
        onClick={() => setPanelVisible(true)}
        aria-label="Show panel"
      >
        <PanelLeftOpenIcon />
      </button>
    </>
  )
}

export default FloatingInfoPanel
