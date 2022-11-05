/* eslint-disable no-console */
import { TSqlTokenizr } from "@/tsql-tokenizr";
import { describe, expect, test } from "@jest/globals";

describe("tsql", () => {
  test("tsql to tokens", () => {
    const tokenizr = new TSqlTokenizr();
    const result = tokenizr
      .tokens(
        "SELECT tb1.id AS id, 'flash' AS name FROM myUser AS tb1 WITH(NOLOCK)"
      )
      .map((x) => x.text);
    expect(result).toStrictEqual([
      "SELECT",
      " ",
      "tb1",
      ".",
      "id",
      " ",
      "AS",
      " ",
      "id",
      ",",
      " ",
      "'flash'",
      " ",
      "AS",
      " ",
      "name",
      " ",
      "FROM",
      " ",
      "myUser",
      " ",
      "AS",
      " ",
      "tb1",
      " ",
      "WITH",
      "(",
      "NOLOCK",
      ")",
    ]);
  });
});
