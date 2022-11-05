/* eslint-disable no-console */
import fs from "fs";
import {
  linqIndexListToStrList,
  linqStrListToString,
  linqTokensToIndexList,
} from "@/linq-encoder";
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

    const strList = linqIndexListToStrList(values);
    console.log(strList);

    console.log(linqStrListToString(strList));

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
