import type { ReactNode } from 'react'
import { useEffect } from 'react'
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
  type ThreadMessage,
  type ThreadUserMessagePart,
} from '@assistant-ui/react'
import { localAttachmentAdapter } from './lib/localAttachmentAdapter'
import { agentRespond, getAgentPrompt } from './lib/backendClient'

const extractLastUserMessage = (messages: readonly ThreadMessage[]) => {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'user') return messages[i]
  }
  return undefined
}

const extractTextFromParts = (parts: readonly ThreadUserMessagePart[]) => {
  return parts
    .map((part) => (part.type === 'text' ? part.text : ''))
    .filter((text) => text.trim().length > 0)
    .join('\n')
}

const extractImageBase64 = (parts: readonly ThreadUserMessagePart[]) => {
  const imagePart = parts.find((part) => part.type === 'image')
  if (!imagePart || !('image' in imagePart)) return undefined
  const image = imagePart.image
  if (typeof image !== 'string') return undefined
  if (image.startsWith('data:')) {
    const comma = image.indexOf(',')
    return comma >= 0 ? image.slice(comma + 1) : undefined
  }
  return image
}

const qwenModelAdapter: ChatModelAdapter = {
  async run(options) {
    const lastUser = extractLastUserMessage(options.messages)
    const parts = (lastUser?.content ?? []) as ThreadUserMessagePart[]
    const instruction = extractTextFromParts(parts) || 'Continue'
    const image_base64 = extractImageBase64(parts)

    let text = '{"action":"stop","n_steps":1}'
    try {
      const response = await agentRespond(
        {
          instruction,
          image_base64,
        },
        options.abortSignal,
      )

      text = response.json
        ? JSON.stringify(response.json)
        : response.text || text
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Agent request failed'
      text = `Error: ${message}`
    }

    return {
      content: [
        {
          type: 'text',
          text,
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
  useEffect(() => {
    let mounted = true
    getAgentPrompt()
      .then((result) => {
        if (mounted) {
          ;(globalThis as { __hexyAgentPrompt?: string }).__hexyAgentPrompt =
            result.prompt
        }
      })
      .catch(() => {
        return
      })
    return () => {
      mounted = false
    }
  }, [])

  const runtime = useLocalRuntime(qwenModelAdapter, {
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
