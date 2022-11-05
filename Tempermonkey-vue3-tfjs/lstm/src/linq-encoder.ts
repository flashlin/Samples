import { keywords } from "@/linq-tokenizr";
import { Token } from "ts-tokenizr";

const linqCharacters = [
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
  return charToIndexDict[ch];
}

function linqTokenToValues(token: Token): number[] {
  const typeIndex = charToIndex(token.type);
  if (token.type == "keyword") {
    return [typeIndex, charToIndexDict[token.text]];
  }
  const next = [...token.text].map((x) => charToIndex(x));
  return [typeIndex, ...next, 0];
}

export function linqTokensToIndexList(tokens: Token[]): number[] {
  const begin = charToIndexDict["<begin>"];
  const end = charToIndexDict["<end>"];
  const body = tokens.map((x) => linqTokenToValues(x)).flatMap((x) => x);
  return [begin, ...body, end];
}

export function linqIndexListToStrList(values: number[]): string[] {
  return values.map((ch) => indexToCharDict[ch]);
}

export function linqStrListToString(strList: string[]): string {
  const process = (acc: string[], arr: string[]): string[] => {
    if (arr.length == 0) {
      return [];
    }
    const first = arr[0];
    if (first.startsWith("<") && arr[0].endsWith(">")) {
      return process([], arr.slice(1));
    }
    if (first == "keyword") {
      return [arr[1], ...process([], arr.slice(2))];
    }
    if (["identifier", "symbol"].includes(first)) {
      return [...process([...acc, arr[1]], arr.slice(2))];
    }
    if (first == "") {
      return [acc.join(""), ...process([], arr.slice(1))];
    }
    return process([...acc, first], arr.slice(1));
  };
  return process([], strList).join(" ");
}
