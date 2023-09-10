import { keywords, TSqlTokenizr } from "@/tsql-tokenizr";
import { ArrayToChar2IndexMap, ArrayToIndex2CharMap } from "@/tokenizr-utils";
import { Token } from "ts-tokenizr";

const tsqlReverseCharacters = [
  "",
  "<begin>",
  "<end>",
  "keyword",
  "operator",
  "symbol",
  "identifier",
  "string",
  "spaces",
];

export const tsqlCharacters = [
  ...tsqlReverseCharacters,
  ...keywords,
  ..."abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`1234567890-=~!@#$%^&*()_+{}|[]\\:\";'<>?,./ ",
];

const charToIndexDict = ArrayToChar2IndexMap(tsqlCharacters);

const indexToCharDict = ArrayToIndex2CharMap(tsqlCharacters);

function charToIndex(ch: string) {
  if (!charToIndexDict.has(ch)) {
    throw new Error(`'${ch}' not exist in dict`);
  }
  return charToIndexDict.get(ch);
}

function tsqlTokenToValues(token: Token): number[] {
  const typeIndex = charToIndex(token.type);
  if (token.type == "keyword") {
    return [typeIndex, charToIndexDict.get(token.text)];
  }
  const next = [...(token.value as string)].map((ch) => charToIndex(ch));
  return [typeIndex, ...next, 0];
}

export function tsqlTokensToIndexList(tokens: Token[]): number[] {
  const begin = charToIndexDict.get("<begin>");
  const end = charToIndexDict.get("<end>");
  const body = tokens.map((x) => tsqlTokenToValues(x)).flatMap((x) => x);
  return [begin, ...body, end];
}

export function tsqlIndexListToString(values: number[]): string {
  const strList = values
    .map((ch) => indexToCharDict.get(ch))
    .filter((ch) => !tsqlReverseCharacters.includes(ch));
  return strList.join("");
}

export function tsqlToIndexList(sql: string): number[] {
  const tokenizr = new TSqlTokenizr();
  const tokens = tokenizr.tokens(sql);
  return tsqlTokensToIndexList(tokens);
}
