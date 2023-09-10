import { keywords, LinqTokenizr } from "@/linq-tokenizr";
import { Token } from "ts-tokenizr";

const linqReverseCharacters = [
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

export const linqCharacters = [
  ...linqReverseCharacters,
  ...keywords,
  ..."abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`1234567890-=~!@#$%^&*()_+{}|[]\\:\";'<>?,./ ",
];

const charToIndexDict = new Map();
linqCharacters.forEach((ch, idx) => {
  charToIndexDict.set(ch, idx);
});

const indexToCharDict = new Map();
linqCharacters.forEach((ch, idx) => {
  indexToCharDict.set(idx, ch);
});

function charToIndex(ch: string) {
  if (!charToIndexDict.has(ch)) {
    throw new Error(`'${ch}' not exist in dict`);
  }
  return charToIndexDict.get(ch);
}

function linqTokenToValues(token: Token): number[] {
  const typeIndex = charToIndex(token.type);
  if (token.type == "keyword") {
    return [typeIndex, charToIndex(token.text)];
  }
  const next = [...(token.value as string)].map((x) => charToIndex(x));
  return [typeIndex, ...next, 0];
}

export function linqTokensToIndexList(tokens: Token[]): number[] {
  const begin = charToIndexDict.get("<begin>");
  const end = charToIndexDict.get("<end>");
  const body = tokens.map((x) => linqTokenToValues(x)).flatMap((x) => x);
  return [begin, ...body, end];
}

export function linqIndexListToString(values: number[]): string {
  const strList = values
    .map((ch) => indexToCharDict.get(ch))
    .filter((ch) => !linqReverseCharacters.includes(ch));
  return strList.join("");
}

export function linqStringToIndexList(expression: string): number[] {
  const tokenizr = new LinqTokenizr();
  const tokens = tokenizr.tokens(expression);
  return linqTokensToIndexList(tokens);
}
