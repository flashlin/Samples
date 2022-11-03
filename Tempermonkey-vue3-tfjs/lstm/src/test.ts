import fs from "fs";
import { LinqTokenizr } from "linq-tokenizr";

let linqLexer = new LinqTokenizr();

let text = fs.readFileSync("./data/sample-sql.txt", "utf8");
text.split("\n").forEach((line, idx) => {
  if (idx % 2 == 0) {
    linqLexer.tokens(line).forEach((token) => {
      console.log(" => ", token.toString());
    });
    console.log(" ");
    return;
  }
});
