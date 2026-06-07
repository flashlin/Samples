import { z } from "zod"
import type { BrowserMethod } from "./methods"

export const tabIdSchema = z.number().int().positive()

export const refTargetSchema = z.object({
  tabId: tabIdSchema,
  snapshotId: z.string().min(1),
  ref: z.string().regex(/^@?e\d+$/)
})

export const snapshotRequestSchema = z.object({
  tabId: tabIdSchema,
  interactiveOnly: z.boolean().optional(),
  compact: z.boolean().optional(),
  maxDepth: z.number().int().min(0).max(100).optional(),
  selector: z.string().min(1).optional(),
  includeUrls: z.boolean().optional(),
  includeIframes: z.boolean().optional(),
  includeCursorInteractive: z.boolean().optional(),
  maxChars: z.number().int().positive().max(1_000_000).optional()
})

export const tabListParamsSchema = z.object({})
export const tabGetActiveParamsSchema = z.object({})
export const pageMetadataParamsSchema = z.object({ tabId: tabIdSchema })
export const pageTextParamsSchema = z.object({
  tabId: tabIdSchema,
  selector: z.string().min(1).optional(),
  maxChars: z.number().int().positive().max(1_000_000).optional()
})
export const pageHtmlParamsSchema = z.object({
  tabId: tabIdSchema,
  selector: z.string().min(1).optional(),
  maxBytes: z.number().int().positive().max(5_000_000).optional(),
  sanitize: z.boolean().optional()
})
export const pageNavigateParamsSchema = z.object({
  tabId: tabIdSchema,
  url: z.url()
})
export const pageReloadParamsSchema = z.object({
  tabId: tabIdSchema,
  ignoreCache: z.boolean().optional()
})
export const elementClickParamsSchema = refTargetSchema
export const elementFillParamsSchema = refTargetSchema.extend({ value: z.string() })
export const elementTypeParamsSchema = refTargetSchema.extend({ text: z.string() })
export const elementGetTextParamsSchema = refTargetSchema
export const networkStartCaptureParamsSchema = z.object({ tabId: tabIdSchema })
export const networkListRequestsParamsSchema = z.object({
  tabId: tabIdSchema,
  limit: z.number().int().positive().max(500).optional()
})
export const networkGetResponseBodyParamsSchema = z.object({
  tabId: tabIdSchema,
  requestId: z.string().min(1)
})
export const debugGetConsoleParamsSchema = z.object({
  tabId: tabIdSchema,
  limit: z.number().int().positive().max(500).optional()
})
export const debugGetErrorsParamsSchema = debugGetConsoleParamsSchema
export const debugEvaluateParamsSchema = z.object({
  tabId: tabIdSchema,
  expression: z.string().min(1).max(100_000),
  returnByValue: z.boolean().optional()
})
export const policyListAllowedOriginsParamsSchema = z.object({})

export const browserCommandParamSchemas = {
  "tabs.list": tabListParamsSchema,
  "tabs.getActive": tabGetActiveParamsSchema,
  "page.snapshot": snapshotRequestSchema,
  "page.getMetadata": pageMetadataParamsSchema,
  "page.getText": pageTextParamsSchema,
  "page.getHtml": pageHtmlParamsSchema,
  "page.navigate": pageNavigateParamsSchema,
  "page.reload": pageReloadParamsSchema,
  "element.click": elementClickParamsSchema,
  "element.fill": elementFillParamsSchema,
  "element.type": elementTypeParamsSchema,
  "element.getText": elementGetTextParamsSchema,
  "network.startCapture": networkStartCaptureParamsSchema,
  "network.listRequests": networkListRequestsParamsSchema,
  "network.getResponseBody": networkGetResponseBodyParamsSchema,
  "debug.getConsole": debugGetConsoleParamsSchema,
  "debug.getErrors": debugGetErrorsParamsSchema,
  "debug.evaluate": debugEvaluateParamsSchema,
  "policy.listAllowedOrigins": policyListAllowedOriginsParamsSchema
} satisfies Record<BrowserMethod, z.ZodType>

export type BrowserCommandParams = {
  [TMethod in BrowserMethod]: z.infer<(typeof browserCommandParamSchemas)[TMethod]>
}

export type RefTarget = z.infer<typeof refTargetSchema>
export type SnapshotRequest = z.infer<typeof snapshotRequestSchema>
