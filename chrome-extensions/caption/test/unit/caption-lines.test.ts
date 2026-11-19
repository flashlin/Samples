import { describe, expect, it } from "vitest";
import { latestSentences, previousCue, splitIntoSentences } from "../../src/shared/cue/caption-lines";

describe("splitIntoSentences", () => {
  it.each([
    [null, []],
    ["", []],
    ["我叫許嵐，做了八年專業管家。", ["我叫許嵐，做了八年專業管家。"]],
    ["第一句。第二句！第三句？", ["第一句。", "第二句！", "第三句？"]],
    ["It works. Really? Yes!", ["It works.", "Really?", "Yes!"]],
    ["version 1.5 is out", ["version 1.5 is out"]],
    ["Today we talk about attention\n今天我們討論 attention。", ["Today we talk about attention\n今天我們討論 attention。"]],
    ["第一行\n第二行", ["第一行", "第二行"]],
  ])("splits %j", (text, expected) => {
    expect(splitIntoSentences(text)).toEqual(expected);
  });
});

describe("latestSentences", () => {
  it("shows only the newest sentence by default", () => {
    expect(latestSentences("第一句。", "第二句。", 1)).toBe("第二句。");
    expect(latestSentences(null, "做了八年。三年前林小姐請我來。", 1)).toBe("三年前林小姐請我來。");
  });

  it("keeps an English sentence and its translation together", () => {
    expect(latestSentences("上一句。", "Hello there\n你好", 1)).toBe("Hello there\n你好");
  });

  it("can show more sentences when configured", () => {
    expect(latestSentences("第一句。", "第二句。第三句。", 2)).toBe("第二句。\n第三句。");
  });

  it("returns null for empty input", () => {
    expect(latestSentences(null, null, 1)).toBeNull();
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
