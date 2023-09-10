/* eslint-disable no-console */
import fs from "fs";
import { linqIndexListToString, linqTokensToIndexList } from "@/linq-encoder";
import { LinqTokenizr } from "@/linq-tokenizr";
import { TSqlTokenizr } from "@/sql-tokenizr";
import { tsqlTokensToIndexList } from "@/tsql-encoder";

const linqLexer = new LinqTokenizr();
const tsqlLexer = new TSqlTokenizr();
const text = fs.readFileSync("./data/sample-sql.txt", "utf8");
text.split("\n").forEach((line, idx) => {
  if (idx % 2 == 0) {
    const tokens = linqLexer.tokens(line);
    const values = linqTokensToIndexList(tokens);
    console.log(values);
    console.log(linqIndexListToString(values));
    console.log(" ");
    return;
  }

  {
    console.log("TSQL:", line);
    const tokens = tsqlLexer.tokens(line);
    const values = tsqlTokensToIndexList(tokens);
    console.log(values);
  }
});
