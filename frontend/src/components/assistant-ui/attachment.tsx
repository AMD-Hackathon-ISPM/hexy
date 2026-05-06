"use client";

import {
  type PropsWithChildren,
  useEffect,
  useMemo,
  useState,
  type FC,
} from "react";
import { XIcon, PlusIcon, FileText } from "lucide-react";
import {
  AttachmentPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  useAuiState,
  useAui,
} from "@assistant-ui/react";
import { useShallow } from "zustand/shallow";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogTrigger,
} from "@/components/ui/dialog";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { cn } from "@/lib/utils";

type AttachmentMediaKind = "image" | "video";

const useFileSrc = (file: File | undefined) => {
  const src = useMemo(() => {
    if (!file) return undefined;
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    if (!src) return;

    return () => {
      URL.revokeObjectURL(src);
    };
  }, [src]);

  return src;
};

const useAttachmentMeta = () => {
  const meta = useAuiState(
    useShallow(
      (s): {
        file?: File;
        src?: string;
        mediaKind?: AttachmentMediaKind;
        name: string;
        typeLabel: string;
      } => {
        const attachment = s.attachment;
        const contentType = attachment.contentType ?? attachment.file?.type ?? "";
        const imagePart = attachment.content?.find((part) => part.type === "image");
        const filePart = attachment.content?.find((part) => part.type === "file");
        const mediaKind =
          attachment.type === "image" ||
          contentType.startsWith("image/") ||
          imagePart
            ? "image"
            : attachment.type === "video" || contentType.startsWith("video/")
              ? "video"
              : undefined;
        const typeLabel =
          mediaKind === "image"
            ? "Image"
            : mediaKind === "video"
              ? "Video"
              : "File";

        return {
          file: attachment.file,
          src: imagePart?.image ?? filePart?.data,
          mediaKind,
          name: attachment.name,
          typeLabel,
        };
      },
    ),
  );
  const fileSrc = useFileSrc(meta.file);

  return { ...meta, src: fileSrc ?? meta.src };
};

const useAttachmentMedia = () => {
  const { src, mediaKind, name } = useAttachmentMeta();

  if (!src || !mediaKind) return undefined;

  return { src, mediaKind, name };
};

type AttachmentPreviewProps = {
  src: string;
  mediaKind: AttachmentMediaKind;
  name: string;
};

const AttachmentPreview: FC<AttachmentPreviewProps> = ({
  src,
  mediaKind,
  name,
}) => {
  const [isLoaded, setIsLoaded] = useState(false);

  if (mediaKind === "video") {
    return (
      <video
        src={src}
        controls
        className="block max-h-[80vh] max-w-full"
        aria-label={name}
      />
    );
  }

  return (
    <img
      src={src}
      alt={name}
      className={cn(
        "block h-auto max-h-[80vh] w-auto max-w-full object-contain",
        isLoaded
          ? "aui-attachment-preview-image-loaded"
          : "aui-attachment-preview-image-loading invisible",
      )}
      onLoad={() => setIsLoaded(true)}
    />
  );
};

const AttachmentPreviewDialog: FC<PropsWithChildren> = ({ children }) => {
  const media = useAttachmentMedia();

  if (!media) return children;

  return (
    <Dialog>
      <DialogTrigger
        className="aui-attachment-preview-trigger cursor-pointer transition-colors hover:bg-accent/50"
        asChild
      >
        {children}
      </DialogTrigger>
      <DialogContent className="aui-attachment-preview-dialog-content p-2 sm:max-w-3xl [&>button]:rounded-full [&>button]:bg-foreground/60 [&>button]:p-1 [&>button]:opacity-100 [&>button]:ring-0! [&_svg]:text-background [&>button]:hover:[&_svg]:text-destructive">
        <DialogTitle className="aui-sr-only sr-only">
          {media.mediaKind === "video" ? "Video" : "Image"} Attachment Preview
        </DialogTitle>
        <div className="aui-attachment-preview relative mx-auto flex max-h-[80dvh] w-full items-center justify-center overflow-hidden bg-background">
          <AttachmentPreview {...media} />
        </div>
      </DialogContent>
    </Dialog>
  );
};

const AttachmentThumb: FC = () => {
  const { src, mediaKind, name } = useAttachmentMeta();

  if (mediaKind === "image" && src) {
    return <img src={src} alt={name} className="hexy-attachment-thumb-media" />;
  }

  if (mediaKind === "video" && src) {
    return (
      <video
        src={src}
        className="hexy-attachment-thumb-media"
        muted
        playsInline
        preload="metadata"
        aria-label={name}
      />
    );
  }

  return (
    <span className="hexy-attachment-thumb-file">
      <FileText />
    </span>
  );
};

const AttachmentUI: FC = () => {
  const aui = useAui();
  const isComposer = aui.attachment.source !== "message";
  const { mediaKind, name, typeLabel } = useAttachmentMeta();

  return (
    <Tooltip>
      <AttachmentPrimitive.Root
        className="hexy-attachment-root"
        data-composer={isComposer ? "true" : "false"}
        data-media={mediaKind ?? "file"}
      >
        <AttachmentPreviewDialog>
          <TooltipTrigger asChild>
            <div
              className="hexy-attachment-chip"
              role="button"
              tabIndex={0}
              aria-label={`${typeLabel} attachment`}
            >
              <AttachmentThumb />
              <span className="hexy-attachment-name">{name}</span>
            </div>
          </TooltipTrigger>
        </AttachmentPreviewDialog>
        {isComposer && <AttachmentRemove />}
      </AttachmentPrimitive.Root>
      <TooltipContent side="top">{name}</TooltipContent>
    </Tooltip>
  );
};

const AttachmentRemove: FC = () => {
  return (
    <AttachmentPrimitive.Remove asChild>
      <TooltipIconButton
        tooltip="Remove file"
        className="hexy-attachment-remove"
        side="top"
      >
        <XIcon />
      </TooltipIconButton>
    </AttachmentPrimitive.Remove>
  );
};

export const UserMessageAttachments: FC = () => {
  return (
    <div className="aui-user-message-attachments-end col-span-full col-start-1 row-start-1 flex w-full flex-row justify-end gap-2">
      <MessagePrimitive.Attachments>
        {() => <AttachmentUI />}
      </MessagePrimitive.Attachments>
    </div>
  );
};

export const ComposerAttachments: FC = () => {
  return (
    <div className="aui-composer-attachments flex w-full flex-row items-center gap-2 overflow-x-auto empty:hidden">
      <ComposerPrimitive.Attachments>
        {() => <AttachmentUI />}
      </ComposerPrimitive.Attachments>
    </div>
  );
};

export const ComposerAddAttachment: FC = () => {
  return (
    <ComposerPrimitive.AddAttachment asChild>
      <TooltipIconButton
        tooltip="Add Attachment"
        side="bottom"
        variant="ghost"
        size="icon"
        className="aui-composer-add-attachment size-8 rounded-full p-1 font-semibold text-xs hover:bg-muted-foreground/15 dark:border-muted-foreground/15 dark:hover:bg-muted-foreground/30"
        aria-label="Add Attachment"
      >
        <PlusIcon className="aui-attachment-add-icon size-5 stroke-[1.5px]" />
      </TooltipIconButton>
    </ComposerPrimitive.AddAttachment>
  );
};
