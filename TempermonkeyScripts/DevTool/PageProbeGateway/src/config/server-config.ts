import { z } from "zod"

const serverConfigSchema = z.object({
  hostname: z.literal("127.0.0.1").default("127.0.0.1"),
  port: z.coerce.number().int().positive().max(65_535).default(17_890),
  tokenPath: z.string().min(1).default(".data/token"),
  requestTimeoutMs: z.coerce.number().int().positive().default(15_000),
  keepAliveIntervalMs: z.coerce.number().int().positive().default(20_000),
  allowedOrigins: z.array(z.string()).default([]),
  allowedHosts: z.array(z.string()).default(["127.0.0.1:17890", "localhost:17890"])
})

export type ServerConfig = z.infer<typeof serverConfigSchema>

export function loadServerConfig(environment: Record<string, string | undefined> = process.env): ServerConfig {
  return serverConfigSchema.parse({
    hostname: environment.PAGE_PROBE_HOST,
    port: environment.PAGE_PROBE_PORT,
    tokenPath: environment.PAGE_PROBE_TOKEN_PATH,
    requestTimeoutMs: environment.PAGE_PROBE_REQUEST_TIMEOUT_MS,
    keepAliveIntervalMs: environment.PAGE_PROBE_KEEPALIVE_MS,
    allowedOrigins: parseList(environment.PAGE_PROBE_ALLOWED_ORIGINS),
    allowedHosts: parseList(environment.PAGE_PROBE_ALLOWED_HOSTS)
  })
}

function parseList(value: string | undefined): string[] | undefined {
  return value?.split(",").map((entry) => entry.trim()).filter(Boolean)
}
