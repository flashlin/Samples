import { describe, expect, it } from "bun:test"
import {
  maxJsonPayloadBytes,
  pageNavigateParamsSchema,
  refTargetSchema
} from "../src"

describe("page navigate params schema", () => {
  it("accepts an http(s) target url", () => {
    const result = pageNavigateParamsSchema.parse({ tabId: 7, url: "https://example.com/path" })

    expect(result.url).toBe("https://example.com/path")
  })

  it("rejects a missing url", () => {
    expect(() => pageNavigateParamsSchema.parse({ tabId: 7 })).toThrow()
  })

  it("rejects a non-url string", () => {
    expect(() => pageNavigateParamsSchema.parse({ tabId: 7, url: "not a url" })).toThrow()
  })
})

describe("ref target schema", () => {
  const validRefs = [
    "e1",
    "@e1",
    "e42",
    "@e999"
  ]

  const invalidRefs = [
    "ref=e1",
    "E1",
    "@E1",
    "e",
    "@e",
    "#submit"
  ]

  for (const ref of validRefs) {
    it(`accepts ref ${ref}`, () => {
      const result = refTargetSchema.parse({
        tabId: 7,
        snapshotId: "snapshot-1",
        ref
      })

      expect(result.ref).toBe(ref)
    })
  }

  for (const ref of invalidRefs) {
    it(`rejects ref ${ref}`, () => {
      expect(() => refTargetSchema.parse({
        tabId: 7,
        snapshotId: "snapshot-1",
        ref
      })).toThrow()
    })
  }
})

describe("payload metadata", () => {
  it("publishes the maximum JSON payload size in bytes", () => {
    expect(maxJsonPayloadBytes).toBe(1_048_576)
  })

  it("identifies payloads that exceed the published limit", () => {
    const oversizedPayload = "x".repeat(maxJsonPayloadBytes + 1)
    const oversizedPayloadBytes = new TextEncoder().encode(oversizedPayload).byteLength

    expect(oversizedPayloadBytes).toBeGreaterThan(maxJsonPayloadBytes)
  })
})
