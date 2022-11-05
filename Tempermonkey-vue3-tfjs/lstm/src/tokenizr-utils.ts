export function keywordsRegExp(arr: string[], map: Function = (x) => x) {
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
