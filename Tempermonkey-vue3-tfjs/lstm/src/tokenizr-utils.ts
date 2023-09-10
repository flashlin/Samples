import { CaseInsensitiveMap } from "./utils/CaseInsensitiveMap";
type MapFn = (ch: string) => string;

export function keywordsRegExp(arr: string[], map: MapFn = (x) => x) {
  const newArr = [...arr];
  newArr.sort().reverse();
  const pattern = newArr.map((x) => `(${map(x)})`).join("|");
  return new RegExp(pattern, "i");
}

export function symbolsRegExp(arr: string[]) {
  const newArr = [...arr];
  newArr.sort().reverse();

  const addEscap = (str: string) => {
    return [...str].map((x) => `\\${x}`).join("");
  };

  const pattern = newArr.map((x) => `(${addEscap(x)})`).join("|");
  return new RegExp(pattern, "i");
}

export function ArrayToChar2IndexMap(arr: string[]): Map<string, number> {
  const charToIndexDict = new Map();
  arr.forEach((ch, idx) => {
    charToIndexDict.set(ch, idx);
  });
  return charToIndexDict;
}

export function ArrayToChar2IndexCaseInsensitiveMap(
  arr: string[]
): CaseInsensitiveMap<string, number> {
  const charToIndexDict = new CaseInsensitiveMap<string, number>();
  arr.forEach((ch, idx) => {
    charToIndexDict.set(ch, idx);
  });
  return charToIndexDict;
}

export function ArrayToIndex2CharMap(arr: string[]): Map<number, string> {
  const indexToCharDict = new Map();
  arr.forEach((ch, idx) => {
    indexToCharDict.set(idx, ch);
  });
  return indexToCharDict;
}
