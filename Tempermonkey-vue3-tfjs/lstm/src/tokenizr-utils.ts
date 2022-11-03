export function keywordsRegExp(arr: string[], map: Function = (x) => x) {
  let newArr = [...arr];
  newArr.sort().reverse();
  let pattern = newArr.map((x) => `(${map(x)})`).join("|");
  return new RegExp(pattern, "i");
}

export function symbolsRegExp(arr: string[]) {
  let newArr = [...arr];
  newArr.sort().reverse();

  const addEscap = (str: string) => {
    return [...str].map((x) => `\\${x}`).join("");
  };

  let pattern = newArr.map((x) => `(${addEscap(x)})`).join("|");
  return new RegExp(pattern, "i");
}