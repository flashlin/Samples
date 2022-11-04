import { keywords } from "sql-tokenizr";
import { Token } from "ts-tokenizr";

const tsqlCharacters = [
  "",
  "<begin>",
  "<end>",
  "keyword",
  "operator",
  "symbol",
  "identifier",
  ...keywords,
  ..."abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`1234567890-=~!@#$%^&*()_+{}|[]\\:\";'<>?,./ ",
];

const charToIndexDict = Object.assign(
  {},
  ...tsqlCharacters.map((ch, idx) => ({ [ch]: idx }))
);

const indexToCharDict = Object.assign(
  {},
  ...tsqlCharacters.map((ch, idx) => ({ [idx]: ch }))
);

function charToIndex(ch: string) {
  if (!charToIndexDict.hasOwnProperty(ch)) {
    throw new Error(`'${ch}' not exist in dict`);
  }
  return charToIndexDict[ch];
}

function getDictValueCaseInsensitive(dict, key) {
   const asLowercase = key.toLowerCase();
   return dict[Object.keys(dict)
     .find(k => k.toLowerCase() === asLowercase)
   ];
 }

function tsqlTokenToValues(token: Token): number[] {
  let typeIndex = charToIndex(token.type);
  if (token.type == "keyword") {
    return [typeIndex, getDictValueCaseInsensitive(charToIndexDict,token.text)];
  }
  let next = [...token.text].map((ch) => charToIndex(ch));
  return [typeIndex, ...next, 0];
}

export function tsqlTokensToIndexList(tokens: Token[]): number[] {
  let begin = charToIndexDict["<begin>"];
  let end = charToIndexDict["<end>"];
  let body = tokens.map((x) => tsqlTokenToValues(x)).flatMap((x) => x);
  return [begin, ...body, end];
}
