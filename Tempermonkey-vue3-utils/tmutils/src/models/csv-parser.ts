
export type SelectorFn = <T>(item: T) => string;

declare global {
  interface String {
    csvSplit(separator?: string): string[];
    camelCase(): string;
    toJson(): string;
  }
  interface Array<T> {
    getCsvHeaders(selector?: SelectorFn): T[];
  }
}

String.prototype.csvSplit = function (separator: string='\n') {
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

String.prototype.camelCase = function() {
  return this.substring(0, 1).toUpperCase() + this.substring(1);
}

function dataLinesToJson(columnsName: string[], lines: string[]) {
  let result: object[] = [];
  lines.forEach((line) => {
    let obj: any = {};
    line.csvSplit().forEach((value, idx) => {
      let name = columnsName[idx].camelCase();
      obj[name] = value;
    });
    result.push(obj);
  });
  return JSON.stringify(result);
}

String.prototype.toJson = function (separator: string='\n') {
  let lines = this.csvSplit(separator);
  let columnNames = lines.getCsvHeaders();
  let dataLines = lines.slice(1);
  return dataLinesToJson(columnNames, dataLines);
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
