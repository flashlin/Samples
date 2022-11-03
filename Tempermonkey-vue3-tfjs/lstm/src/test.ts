import fs from "fs";
import { LinqTokenizr } from "linq-tokenizr";
import { Token } from "ts-tokenizr";

let linqLexer = new LinqTokenizr();

const linqCharacters = [
  "<begin>",
  "<end>",
  "keyword",
  "operator",
  "symbol",
  "identifier",
  ..."abcdefghijklmnopqrstuvwxyz`1234567890-=~!@#$%^&*()_+{}|[]\\:\";'<>?,./ ",
];

const indexDict = Object.assign(
  {},
  ...linqCharacters.map((x, idx) => ({ [x]: idx }))
);

const valueDict = Object.assign(
  {},
  ...linqCharacters.map((x, idx) => ({ [idx]: x }))
);

function toIndex(ch: string) {
  if (!indexDict.hasOwnProperty(ch)) {
    throw new Error(`'${ch}' not exist in dict`);
  }
  return indexDict[ch];
}

function tokenToValues(token: Token): number[] {
  let prev = toIndex(token.type);
  let next = [...token.text].map((x) => toIndex(x));
  return [prev, ...next];
}

function tokensToIndexList(tokens: Token[]): number[] {
  let begin = indexDict["<begin>"];
  let end = indexDict["<end>"];
  let body = tokens.map((x) => tokenToValues(x)).flatMap((x) => x);
  return [begin, ...body, end];
}

function indexListToStrList(values: number[]) {
  return values.map((x) => valueDict[x]);
}

let text = fs.readFileSync("./data/sample-sql.txt", "utf8");
text.split("\n").forEach((line, idx) => {
  if (idx % 2 == 0) {
    let tokens = linqLexer.tokens(line);
    tokens.pop();
    let values = tokensToIndexList(tokens);
    console.log(values);
    
    let t = indexListToStrList(values);
    console.log(t);
    
    console.log(" ");
    return;
  }
});
