
export type SelectorFn = <T>(item: T) => string;

declare global {
  interface String {
    csvSplit(separator?: string): string[];
  }
  interface Array<T> {
    getCsvHeaders(selector?: SelectorFn): T[];
  }
}

String.prototype.csvSplit = function (separator: string=',') {
  let reg_exp = new RegExp(
    "(\\" +
      separator +
      '|\\r?\\n|\\r|^)(?:"([^"]*(?:""[^"]*)*)"|([^"\\' +
      separator +
      "\\r\\n]*))",
    "gi"
  );
  let str = this.trim(),
    row = [],
    m = null;

  if (str.match(new RegExp("^\\" + separator))) {
    row.push("");
  }

  while ((m = reg_exp.exec(str))) {
    var m1 = m[1];
    if (m1.length && m1 != separator) {
      row.push(m1);
    }
    var value = m[2] ? m[2].replace(new RegExp('""', "g"), '"') : m[3];
    row.push(value);
  }

  return row;
}

const defaultGetCsvHeaderSelector = <T>(item: T) => {
  let obj: any = item;
  return obj as string;
}

Array.prototype.getCsvHeaders = function<T>(selector: SelectorFn=defaultGetCsvHeaderSelector) {
  let columns: any[] = [];
  let line = this[0];
  line.csvSplit().forEach((item: T, _: number) => {
    columns.push(selector(item));
  });
  return columns;
}
