import type { ReactNode } from 'react'
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
} from '@assistant-ui/react'
import { localAttachmentAdapter } from './lib/localAttachmentAdapter'

const mockModelAdapter: ChatModelAdapter = {
  async run() {
    return {
      content: [
        {
          type: 'text',
          text: 'This is a mock assistant response. Backend integration is not connected yet.',
        },
      ],
    }
  },
}

export function AssistantChatRuntimeProvider({
  children,
}: {
  children: ReactNode
}) {
  const runtime = useLocalRuntime(mockModelAdapter, {
    adapters: {
      attachments: localAttachmentAdapter,
    },
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  )
}
