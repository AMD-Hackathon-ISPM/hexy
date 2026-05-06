import type {
  Attachment,
  AttachmentAdapter,
  CompleteAttachment,
  PendingAttachment,
  ThreadUserMessagePart,
} from "@assistant-ui/react";

const readAsDataURL = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

const getAttachmentType = (file: File) => {
  if (file.type.startsWith("image/")) return "image";
  if (file.type.startsWith("video/")) return "video";
  return "file";
};

class LocalAttachmentAdapter implements AttachmentAdapter {
  accept = "*";

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    return {
      id: `${file.name}-${file.size}-${file.lastModified}`,
      type: getAttachmentType(file),
      name: file.name,
      contentType: file.type || "application/octet-stream",
      file,
      status: { type: "requires-action", reason: "composer-send" },
    };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const data = await readAsDataURL(attachment.file);
    const contentType =
      attachment.contentType ||
      attachment.file.type ||
      "application/octet-stream";
    const content: ThreadUserMessagePart[] =
      attachment.type === "image"
        ? [{ type: "image", image: data, filename: attachment.name }]
        : [
            {
              type: "file",
              data,
              mimeType: contentType,
              filename: attachment.name,
            },
          ];

    return {
      ...attachment,
      status: { type: "complete" },
      content,
    };
  }

  async remove(_attachment: Attachment) {}
}

export const localAttachmentAdapter = new LocalAttachmentAdapter();
