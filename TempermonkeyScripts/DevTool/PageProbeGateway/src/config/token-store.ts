import { chmod, mkdir } from "node:fs/promises"
import { dirname, resolve } from "node:path"

const tokenByteLength = 32

export class TokenStore {
  readonly tokenPath: string

  constructor(tokenPath: string) {
    this.tokenPath = resolve(tokenPath)
  }

  async load(environmentToken?: string): Promise<string> {
    if (environmentToken) {
      return environmentToken
    }

    const tokenFile = Bun.file(this.tokenPath)
    if (await tokenFile.exists()) {
      return (await tokenFile.text()).trim()
    }

    const token = createToken()
    await mkdir(dirname(this.tokenPath), { recursive: true })
    await Bun.write(this.tokenPath, token)
    await chmod(this.tokenPath, 0o600)
    return token
  }
}

function createToken(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(tokenByteLength))
  return Buffer.from(bytes).toString("hex")
}
