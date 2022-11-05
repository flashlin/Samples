import { LinqTokenizr } from "@/linq-tokenizr";
import { describe, expect, test } from "@jest/globals";

describe("sum module", () => {
  test("adds 1 + 2 to equal 3", () => {
    const tokenizr = new LinqTokenizr();
    const result = tokenizr
      .tokens('from tb1 in myUser select new{id=tb1.id,name="flash"}')
      .map((x) => x.text);
    expect(result).toBe([
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
