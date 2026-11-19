import { describe, expect, it } from "vitest";
import { previousCue, rollingCaption, splitIntoLines } from "../../src/shared/cue/caption-lines";

describe("splitIntoLines", () => {
  it.each([
    [null, []],
    ["", []],
    ["我叫許嵐，做了八年專業管家。", ["我叫許嵐，做了八年專業管家。"]],
    ["第一句。第二句！第三句？", ["第一句。", "第二句！", "第三句？"]],
    ["It works. Really? Yes!", ["It works.", "Really?", "Yes!"]],
    ["version 1.5 is out", ["version 1.5 is out"]],
    ["Today we talk about attention\n今天我們討論 attention。", ["Today we talk about attention", "今天我們討論 attention。"]],
  ])("splits %j", (text, expected) => {
    expect(splitIntoLines(text)).toEqual(expected);
  });
});

describe("rollingCaption", () => {
  it("puts the new sentence below the previous one", () => {
    expect(rollingCaption("第一句。", "第二句。", 2)).toBe("第一句。\n第二句。");
  });

  it("keeps only the newest lines when there are more than the limit", () => {
    expect(rollingCaption("第一句。", "第二句。第三句。", 2)).toBe("第二句。\n第三句。");
  });

  it("gives a bilingual sentence the whole box", () => {
    expect(rollingCaption("上一句。", "Hello there\n你好", 2)).toBe("Hello there\n你好");
  });

  it("shows a single sentence alone and nothing for empty input", () => {
    expect(rollingCaption(null, "只有一句", 2)).toBe("只有一句");
    expect(rollingCaption(null, null, 2)).toBeNull();
  });
});

describe("previousCue", () => {
  const cues = [
    { start: 0, end: 3, text: "a" },
    { start: 3, end: 6, text: "b" },
    { start: 20, end: 22, text: "c" },
  ];

  it("returns the cue right before the current one", () => {
    expect(previousCue(cues, cues[1]!, 6)?.text).toBe("a");
  });

  it("works for a live cue that is not in the list yet", () => {
    expect(previousCue(cues, { start: 7, end: 8, text: "interim" }, 6)?.text).toBe("b");
  });

  it("drops a previous cue that ended too long ago", () => {
    expect(previousCue(cues, cues[2]!, 6)).toBeNull();
    expect(previousCue(cues, cues[0]!, 6)).toBeNull();
  });
});
