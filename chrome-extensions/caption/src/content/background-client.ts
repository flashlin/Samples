import type { ContentRequest, Reply, ResolvedPage, SaveResult, TranslateLinesResult } from "../shared/messages";

interface ResultByRequest {
  "resolve-page": ResolvedPage;
  "translate-lines": TranslateLinesResult;
  "save-subtitle": SaveResult;
  "report-status": null;
  "save-overlay-offset": null;
}

export async function askBackground<R extends ContentRequest>(request: R): Promise<ResultByRequest[R["type"]]> {
  const reply = (await chrome.runtime.sendMessage(request)) as Reply<ResultByRequest[R["type"]]> | undefined;
  if (!reply) {
    throw new Error("No reply from the caption background worker");
  }
  if (!reply.ok) {
    throw new Error(reply.error);
  }
  return reply.value;
}
