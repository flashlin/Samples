import type { ConnectionConfigStore } from "../config/connection-config-store"

export class PolicyService {
  constructor(private readonly configStore: ConnectionConfigStore) {}

  async listAllowedOrigins(): Promise<{ origins: string[] }> {
    const config = await this.configStore.load()
    return { origins: config.allowedOrigins }
  }
}
