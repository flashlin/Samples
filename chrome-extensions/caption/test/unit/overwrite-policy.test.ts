import { describe, expect, it } from "vitest";
import cases from "../fixtures/overwrite-policy-cases.json";
import { decideOverwrite, type OverwriteDecision, type SubtitleProgress } from "../../src/shared/progress/overwrite-policy";

interface PolicyCase {
  name: string;
  existing: SubtitleProgress | null;
  candidate: SubtitleProgress;
  confirmed: boolean;
  expected: OverwriteDecision;
}

describe("decideOverwrite", () => {
  it.each(cases as PolicyCase[])("$name", ({ existing, candidate, confirmed, expected }) => {
    expect(decideOverwrite(existing, candidate, confirmed)).toEqual(expected);
  });
});
