/* eslint-disable no-console */
import { LinqTokenizr } from "@/linq-tokenizr";
import { describe, expect, test } from "@jest/globals";

describe("linq to token", () => {
  test("case1", () => {
    const tokenizr = new LinqTokenizr();
    const result = tokenizr
      .tokens('from tb1 in myUser select new{id=tb1.id,name="flash"}')
      .map((x) => x.text);
    console.info("result=", result);
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
});
