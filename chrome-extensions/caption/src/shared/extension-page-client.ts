import type { ExtensionPageRequest, Reply } from "./messages";

export async function askBackgroundFromPage<T>(request: ExtensionPageRequest): Promise<T> {
  const reply = (await chrome.runtime.sendMessage(request)) as Reply<T> | undefined;
  if (!reply) {
    throw new Error("No reply from the caption background worker");
  }
  if (!reply.ok) {
    throw new Error(reply.error);
  }
  return reply.value;
}
