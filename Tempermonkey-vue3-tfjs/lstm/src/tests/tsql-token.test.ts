/* eslint-disable no-console */
import { tsqlIndexListToString, tsqlTokensToIndexList } from "@/tsql-encoder";
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

  test("tsql * to tokens", () => {
    const tokenizr = new TSqlTokenizr();
    const result = tokenizr.tokens("SELECT * FROM myUser").map((x) => x.value);
    expect(result).toStrictEqual([
      "SELECT",
      " ",
      "*",
      " ",
      "FROM",
      " ",
      "myUser",
    ]);
  });

  test("tsql tokens to values", () => {
    const tokenizr = new TSqlTokenizr();
    const tokens = tokenizr.tokens(
      "SELECT tb1.id AS id, 'flash' AS name FROM myUser AS tb1 WITH(NOLOCK)"
    );
    const values = tsqlTokensToIndexList(tokens);
    //console.log(JSON.stringify(values));
    expect(values).toStrictEqual([
      1, 3, 48, 8, 288, 0, 6, 213, 195, 247, 0, 5, 286, 0, 6, 202, 197, 0, 8,
      288, 0, 3, 188, 8, 288, 0, 6, 202, 197, 0, 5, 285, 0, 8, 288, 0, 7, 281,
      199, 205, 194, 212, 201, 281, 0, 8, 288, 0, 3, 188, 8, 288, 0, 6, 207,
      194, 206, 198, 0, 8, 288, 0, 3, 123, 8, 288, 0, 6, 206, 218, 240, 212,
      198, 211, 0, 8, 288, 0, 3, 188, 8, 288, 0, 6, 213, 195, 247, 0, 8, 288, 0,
      3, 11, 5, 268, 0, 6, 233, 234, 231, 234, 222, 230, 0, 5, 269, 0, 2,
    ]);
  });

  test("tsql token values to string", () => {
    const tokenizr = new TSqlTokenizr();
    const tokens = tokenizr.tokens(
      "SELECT tb1.id AS id, 'flash' AS name FROM myUser AS tb1 WITH(NOLOCK)"
    );
    const values = tsqlTokensToIndexList(tokens);
    const result = tsqlIndexListToString(values);
    expect(result).toBe(
      "SELECT tb1.id AS id, 'flash' AS name FROM myUser AS tb1 WITH(NOLOCK)"
    );
  });
});
