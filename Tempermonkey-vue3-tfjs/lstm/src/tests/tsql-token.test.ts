/* eslint-disable no-console */
import { tsqlTokensToIndexList } from "@/tsql-encoder";
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

  test("tsql tokens to values", () => {
    const tokenizr = new TSqlTokenizr();
    const tokens = tokenizr.tokens(
      "SELECT tb1.id AS id, 'flash' AS name FROM myUser AS tb1 WITH(NOLOCK)"
    );
    const values = tsqlTokensToIndexList(tokens);
    expect(values).toStrictEqual([
      1, 3, 48, 8, 288, 0, 6, 239, 221, 247, 0, 5, 286, 0, 6, 228, 223, 0, 8,
      288, 0, 3, 188, 8, 288, 0, 6, 228, 223, 0, 5, 285, 0, 8, 288, 0, 7, 281,
      225, 231, 220, 238, 227, 281, 0, 8, 288, 0, 3, 188, 8, 288, 0, 6, 233,
      220, 232, 224, 0, 8, 288, 0, 3, 123, 8, 288, 0, 6, 232, 244, 240, 238,
      224, 237, 0, 8, 288, 0, 3, 188, 8, 288, 0, 6, 239, 221, 247, 0, 8, 288, 0,
      3, 11, 5, 268, 0, 6, 233, 234, 231, 234, 222, 230, 0, 5, 269, 0, 2,
    ]);
  });
});
