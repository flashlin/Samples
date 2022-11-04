import fs from "fs";
import { LinqTokenizr, keywords } from "linq-tokenizr";
import { Token } from "ts-tokenizr";

let linqLexer = new LinqTokenizr();

const linqCharacters = [
  "",
  "<begin>",
  "<end>",
  "keyword",
  "operator",
  "symbol",
  "identifier",
  ...keywords,
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
  let typeIndex = toIndex(token.type);
  if (token.type == "keyword") {
    return [typeIndex, indexDict[token.text]];
  }
  let next = [...token.text].map((x) => toIndex(x));
  return [typeIndex, ...next, 0];
}

function tokensToIndexList(tokens: Token[]): number[] {
  let begin = indexDict["<begin>"];
  let end = indexDict["<end>"];
  let body = tokens.map((x) => tokenToValues(x)).flatMap((x) => x);
  return [begin, ...body, end];
}

function indexListToStrList(values: number[]): string[] {
  return values.map((x) => valueDict[x]);
}

function strListToText(strList: string[]): string {
  const process = (acc: string[], arr: string[]): string[] => {
    if( arr.length == 0) {
      return [];
    }
    let first = arr[0];
    if( first.startsWith('<') && arr[0].endsWith('>') ) {
      return process([], arr.slice(1));
    }
    if( first == "keyword" ) {
      return [arr[1], ...process([], arr.slice(2))];
    }
    if( ['identifier', 'symbol'].includes(first) ) {
      return [...process([...acc, arr[1]], arr.slice(2))];
    }
    if( first == '') {
      return [acc.join(''), ...process([], arr.slice(1))];
    }
    return process([...acc, first], arr.slice(1));
  };
  return process([], strList).join(' ');
}

let text = fs.readFileSync("./data/sample-sql.txt", "utf8");
text.split("\n").forEach((line, idx) => {
  if (idx % 2 == 0) {
    let tokens = linqLexer.tokens(line);
    tokens.pop();
    let values = tokensToIndexList(tokens);
    console.log(values);

    let strList = indexListToStrList(values);
    console.log(strList);

    console.log(strListToText(strList));

    console.log(" ");
    return;
  }
});
