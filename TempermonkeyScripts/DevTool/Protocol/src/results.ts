import { z } from "zod"
import type { BrowserMethod } from "./methods"

export const tabInfoSchema = z.object({
  id: z.number().int().positive(),
  active: z.boolean(),
  title: z.string(),
  url: z.string(),
  status: z.string().optional()
})

export const pageMetadataSchema = z.object({
  tabId: z.number().int().positive(),
  title: z.string(),
  url: z.string(),
  origin: z.string()
})

export const publicRefSchema = z.object({
  role: z.string(),
  name: z.string(),
  nth: z.number().int().nonnegative().optional(),
  frameId: z.string().optional()
})

export const snapshotResultSchema = z.object({
  snapshotId: z.string().min(1),
  tabId: z.number().int().positive(),
  documentEpoch: z.number().int().nonnegative(),
  origin: z.string(),
  text: z.string(),
  refs: z.record(z.string(), publicRefSchema),
  truncated: z.boolean()
})

export const textResultSchema = z.object({
  text: z.string(),
  truncated: z.boolean()
})

export const htmlResultSchema = z.object({
  html: z.string(),
  truncated: z.boolean()
})

export const actionResultSchema = z.object({ success: z.literal(true) })

export const networkRequestSchema = z.object({
  requestId: z.string(),
  url: z.string(),
  method: z.string(),
  type: z.string().optional(),
  status: z.number().optional(),
  mimeType: z.string().optional(),
  failed: z.boolean().optional(),
  errorText: z.string().optional(),
  timestamp: z.number(),
  finished: z.boolean()
})

export const responseBodyResultSchema = z.object({
  body: z.string(),
  base64Encoded: z.boolean(),
  truncated: z.boolean(),
  originalSize: z.number().int().nonnegative()
})

export const debugEntrySchema = z.object({
  level: z.string(),
  text: z.string(),
  timestamp: z.number(),
  source: z.string().optional(),
  url: z.string().optional(),
  lineNumber: z.number().int().nonnegative().optional()
})

export const debugEvaluateResultSchema = z.object({
  value: z.unknown().optional(),
  type: z.string(),
  description: z.string().optional()
})

export const allowedOriginsResultSchema = z.object({
  origins: z.array(z.string())
})

export const browserCommandResultSchemas = {
  "tabs.list": z.object({ tabs: z.array(tabInfoSchema) }),
  "tabs.getActive": tabInfoSchema,
  "page.snapshot": snapshotResultSchema,
  "page.getMetadata": pageMetadataSchema,
  "page.getText": textResultSchema,
  "page.getHtml": htmlResultSchema,
  "page.navigate": actionResultSchema,
  "page.reload": actionResultSchema,
  "page.goBack": actionResultSchema,
  "page.goForward": actionResultSchema,
  "element.click": actionResultSchema,
  "element.fill": actionResultSchema,
  "element.type": actionResultSchema,
  "element.getText": textResultSchema,
  "network.startCapture": actionResultSchema,
  "network.listRequests": z.object({ requests: z.array(networkRequestSchema) }),
  "network.getResponseBody": responseBodyResultSchema,
  "debug.getConsole": z.object({ entries: z.array(debugEntrySchema) }),
  "debug.getErrors": z.object({ entries: z.array(debugEntrySchema) }),
  "debug.evaluate": debugEvaluateResultSchema,
  "policy.listAllowedOrigins": allowedOriginsResultSchema
} satisfies Record<BrowserMethod, z.ZodType>

export type BrowserCommandResults = {
  [TMethod in BrowserMethod]: z.infer<(typeof browserCommandResultSchemas)[TMethod]>
}

export type PublicRef = z.infer<typeof publicRefSchema>
export type SnapshotResult = z.infer<typeof snapshotResultSchema>
export type TabInfo = z.infer<typeof tabInfoSchema>
export type NetworkRequest = z.infer<typeof networkRequestSchema>
export type DebugEntry = z.infer<typeof debugEntrySchema>
