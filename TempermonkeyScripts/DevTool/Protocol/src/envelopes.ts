import { z } from "zod"
import { browserCommandParamSchemas, type BrowserCommandParams } from "./commands"
import { browserErrorSchema } from "./errors"
import { browserMethods, type BrowserMethod } from "./methods"
import { browserCommandResultSchemas, type BrowserCommandResults } from "./results"

export const protocolVersion = 1 as const
export const maxJsonPayloadBytes = 1_048_576

export type CommandEnvelope<TMethod extends BrowserMethod = BrowserMethod> = {
  protocolVersion: typeof protocolVersion
  type: "command"
  id: string
  clientId: string
  method: TMethod
  params: BrowserCommandParams[TMethod]
  deadline: string
}

export type ResultEnvelope<TMethod extends BrowserMethod = BrowserMethod> = {
  protocolVersion: typeof protocolVersion
  type: "result"
  id: string
  ok: true
  result: BrowserCommandResults[TMethod]
}

export const errorEnvelopeSchema = z.object({
  protocolVersion: z.literal(protocolVersion),
  type: z.literal("result"),
  id: z.string().min(1),
  ok: z.literal(false),
  error: browserErrorSchema
})

export type ErrorEnvelope = z.infer<typeof errorEnvelopeSchema>

export const authMessageSchema = z.object({
  protocolVersion: z.literal(protocolVersion),
  type: z.literal("auth"),
  token: z.string().min(1),
  client: z.object({
    clientId: z.string().min(1),
    name: z.string().min(1),
    version: z.string().min(1)
  })
})

export const authResultSchema = z.object({
  protocolVersion: z.literal(protocolVersion),
  type: z.literal("authResult"),
  ok: z.boolean(),
  error: browserErrorSchema.optional()
})

export const heartbeatSchema = z.object({
  protocolVersion: z.literal(protocolVersion),
  type: z.enum(["ping", "pong"]),
  timestamp: z.number()
})

export function parseCommandEnvelope(input: unknown): CommandEnvelope {
  const base = z.object({
    protocolVersion: z.literal(protocolVersion),
    type: z.literal("command"),
    id: z.string().min(1),
    clientId: z.string().min(1),
    method: z.enum(browserMethods),
    params: z.unknown(),
    deadline: z.string().datetime()
  }).parse(input)

  const params = browserCommandParamSchemas[base.method].parse(base.params)
  return { ...base, params } as CommandEnvelope
}

export function parseResultEnvelope<TMethod extends BrowserMethod>(
  method: TMethod,
  input: unknown
): ResultEnvelope<TMethod> | ErrorEnvelope {
  const base = z.object({
    protocolVersion: z.literal(protocolVersion),
    type: z.literal("result"),
    id: z.string().min(1),
    ok: z.boolean()
  }).passthrough().parse(input)

  if (!base.ok) {
    return errorEnvelopeSchema.parse(input)
  }

  const resultEnvelope = z.object({
    protocolVersion: z.literal(protocolVersion),
    type: z.literal("result"),
    id: z.string().min(1),
    ok: z.literal(true),
    result: browserCommandResultSchemas[method]
  }).parse(input)

  return resultEnvelope as ResultEnvelope<TMethod>
}

export type AuthMessage = z.infer<typeof authMessageSchema>
export type AuthResult = z.infer<typeof authResultSchema>
export type Heartbeat = z.infer<typeof heartbeatSchema>
