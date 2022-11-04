import fs from "fs";
import { linqIndexListToStrList, linqStrListToString, linqTokensToIndexList } from "linq-encoder";
import { LinqTokenizr, keywords } from "linq-tokenizr";

let linqLexer = new LinqTokenizr();
let text = fs.readFileSync("./data/sample-sql.txt", "utf8");
text.split("\n").forEach((line, idx) => {
  if (idx % 2 == 0) {
    let tokens = linqLexer.tokens(line);
    tokens.pop();
    let values = linqTokensToIndexList(tokens);
    console.log(values);

    let strList = linqIndexListToStrList(values);
    console.log(strList);

    console.log(linqStrListToString(strList));

    console.log(" ");
    return;
  }
});


