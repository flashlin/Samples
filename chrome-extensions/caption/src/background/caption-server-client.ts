import type { SaveResult, SubtitleMetaPayload, TranslateLinesResult } from "../shared/messages";

const MILLISECONDS_PER_SECOND = 1000;
const HTTP_NOT_FOUND = 404;
const HTTP_CONFLICT = 409;

export interface ServerConnection {
  readonly endpoint: string;
  readonly token: string;
  readonly timeoutSeconds: number;
}

export interface StoredSubtitle {
  readonly meta: SubtitleMetaPayload & { readonly revision: number };
  readonly srt: string;
}

export interface SubtitleWriteRequest {
  readonly meta: SubtitleMetaPayload;
  readonly srt: string;
  readonly expectedRevision: number;
  readonly confirmOverwrite: boolean;
}

export class CaptionServerError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
  ) {
    super(message);
  }
}

export class CaptionServerClient {
  constructor(private readonly connection: ServerConnection) {}

  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    if (!this.connection.token) {
      throw new CaptionServerError("Caption server token is not configured, open the extension options", null);
    }
    const headers = new Headers(init.headers);
    headers.set("Authorization", `Bearer ${this.connection.token}`);
    try {
      return await fetch(new URL(path, this.connection.endpoint), {
        ...init,
        headers,
        signal: AbortSignal.timeout(this.connection.timeoutSeconds * MILLISECONDS_PER_SECOND),
      });
    } catch (error) {
      throw new CaptionServerError(`Caption server unreachable: ${error instanceof Error ? error.message : String(error)}`, null);
    }
  }

  private async json<T>(response: Response): Promise<T> {
    if (!response.ok) {
      throw new CaptionServerError(`Caption server responded ${response.status}: ${(await response.text()).slice(0, 200)}`, response.status);
    }
    return (await response.json()) as T;
  }

  async health(): Promise<{ status: string; engine: Record<string, string> }> {
    return this.json(await this.request("/health"));
  }

  async translateLines(texts: string[]): Promise<TranslateLinesResult> {
    const body = await this.json<{ displays: string[] }>(
      await this.request("/v1/translations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ texts }) }),
    );
    return { displays: body.displays };
  }

  async getSubtitle(urlKey: string): Promise<StoredSubtitle | null> {
    const response = await this.request(`/v1/subtitles/${urlKey}`);
    if (response.status === HTTP_NOT_FOUND) {
      return null;
    }
    return this.json(response);
  }

  async putSubtitle(urlKey: string, write: SubtitleWriteRequest): Promise<SaveResult> {
    const response = await this.request(`/v1/subtitles/${urlKey}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(write),
    });
    if (response.status === HTTP_CONFLICT) {
      return (await response.json()) as SaveResult;
    }
    return this.json(response);
  }

  async deleteSubtitle(urlKey: string): Promise<void> {
    const response = await this.request(`/v1/subtitles/${urlKey}`, { method: "DELETE" });
    if (!response.ok && response.status !== HTTP_NOT_FOUND) {
      throw new CaptionServerError(`Caption server responded ${response.status}`, response.status);
    }
  }
}
