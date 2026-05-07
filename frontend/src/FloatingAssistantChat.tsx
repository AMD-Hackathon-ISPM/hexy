import { useState } from "react";
import { useAssistantState } from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/thread";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  ChevronDownIcon,
  ChevronRightIcon,
  MessageCircleIcon,
} from "lucide-react";

export function FloatingAssistantChat() {
  const [visible, setVisible] = useState(true);

  const [expanded, setExpanded] = useState(false);
  const isEmpty = useAssistantState((s) => s.thread.isEmpty);
  const hasComposerAttachments = useAssistantState(
    (s) => s.composer.attachments.length > 0,
  );
  const showHeader = !isEmpty;
  const isExpanded = showHeader && expanded;

  return (
    <>
      <div className="pointer-events-none fixed inset-x-0 bottom-4 z-10 flex justify-center px-4">
        <div
          data-expanded={isExpanded}
          data-empty={isEmpty}
          data-has-attachments={hasComposerAttachments}
          data-visible={visible}
          className={cn(
            "hexy-chat-panel pointer-events-auto relative w-[min(720px,calc(100vw-32px))]",
            !showHeader
              ? hasComposerAttachments
                ? "h-[112px]"
                : "h-[64px]"
              : isExpanded
                ? "h-[min(560px,calc(100vh-32px))]"
                : hasComposerAttachments
                  ? "h-[142px]"
                  : "h-[90px]",
          )}
        >
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="hexy-chat-close-btn"
                  onClick={() => setVisible(false)}
                  aria-label="Close chat"
                >
                  <ChevronDownIcon />
                </button>
              </TooltipTrigger>
              <TooltipContent side="left">Hide chat</TooltipContent>
            </Tooltip>
            <div className="hexy-chat-tray" aria-hidden="true" />
            <button
              type="button"
              className="hexy-chat-header"
              onClick={() => setExpanded((value) => !value)}
              aria-expanded={isExpanded}
              aria-hidden={!showHeader}
              tabIndex={showHeader ? 0 : -1}
              disabled={!showHeader}
            >
              <span>Latest turn</span>
              <span className="hexy-chat-header-actions">
                {isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
              </span>
            </button>
            <Thread />
          </TooltipProvider>
        </div>
      </div>
      <button
        type="button"
        className="hexy-chat-fab"
        data-visible={!visible}
        onClick={() => setVisible(true)}
        aria-label="Open chat"
      >
        <MessageCircleIcon />
        <span className="hexy-chat-fab-badge" />
      </button>
    </>
  );
}
