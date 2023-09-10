/* eslint-disable no-console */
import { linqIndexListToString, linqTokensToIndexList } from "@/linq-encoder";
import { LinqTokenizr } from "@/linq-tokenizr";
import { describe, expect, test } from "@jest/globals";

describe("linq to token", () => {
  test("linq to tokens", () => {
    const tokenizr = new LinqTokenizr();
    const result = tokenizr
      .tokens('from tb1 in myUser select new{id=tb1.id,name="flash"}')
      .map((x) => x.text);
    expect(result).toStrictEqual([
      "from",
      " ",
      "tb1",
      " ",
      "in",
      " ",
      "myUser",
      " ",
      "select",
      " ",
      "new",
      "{",
      "id",
      "=",
      "tb1",
      ".",
      "id",
      ",",
      "name",
      "=",
      '"flash"',
      "}",
    ]);
  });

  test("linq cr to tokens", () => {
    const tokenizr = new LinqTokenizr();
    const result = tokenizr.tokens("\n").map((x) => x.value);
    expect(result).toStrictEqual([" "]);
  });

  test("linq tokens to values", () => {
    const tokenizr = new LinqTokenizr();
    const tokens = tokenizr.tokens(
      'from tb1 in myUser select new{id=tb1.id,name="flash"}'
    );
    const result = linqTokensToIndexList(tokens);
    expect(result).toStrictEqual([
      1, 3, 10, 8, 111, 0, 6, 36, 18, 70, 0, 8, 111, 0, 3, 16, 8, 111, 0, 6, 29,
      41, 63, 35, 21, 34, 0, 8, 111, 0, 3, 9, 8, 111, 0, 6, 30, 21, 39, 0, 5,
      95, 0, 6, 25, 20, 0, 5, 81, 0, 6, 36, 18, 70, 0, 5, 109, 0, 6, 25, 20, 0,
      5, 108, 0, 6, 30, 17, 29, 21, 0, 5, 81, 0, 7, 102, 22, 28, 17, 35, 24,
      102, 0, 5, 96, 0, 2,
    ]);
  });

  test("linq values to string", () => {
    const tokenizr = new LinqTokenizr();
    const tokens = tokenizr.tokens(
      'from tb1 in myUser select new{id=tb1.id,name="flash"}'
    );
    const valueList = linqTokensToIndexList(tokens);

    const result = linqIndexListToString(valueList);
    expect(result).toBe(
      'from tb1 in myUser select new{id=tb1.id,name="flash"}'
    );
  });
});
