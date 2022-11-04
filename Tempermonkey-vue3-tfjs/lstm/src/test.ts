import fs from "fs";
import { linqIndexListToStrList, linqStrListToString, linqTokensToIndexList } from "linq-encoder";
import { LinqTokenizr, keywords } from "linq-tokenizr";
import { TSqlTokenizr } from "sql-tokenizr";
import { tsqlTokensToIndexList } from "tsql-encoder";

let linqLexer = new LinqTokenizr();
let tsqlLexer = new TSqlTokenizr();
let text = fs.readFileSync("./data/sample-sql.txt", "utf8");
text.split("\n").forEach((line, idx) => {
  if (idx % 2 == 0) {
    let tokens = linqLexer.tokens(line);
    let values = linqTokensToIndexList(tokens);
    console.log(values);

    let strList = linqIndexListToStrList(values);
    console.log(strList);

    console.log(linqStrListToString(strList));

    console.log(" ");
    return;
  }

  {
    console.log("TSQL:", line);
    let tokens = tsqlLexer.tokens(line);
    let values = tsqlTokensToIndexList(tokens);
    console.log(values);
  }
});


