
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

function group(context: string) {
  return `(?:${context})`;
}

function quotedStringPattern(symbol: string) {
  const p = group(`\\\\[\\S\\s][^${symbol}\\\\]*`);
  return `${symbol}[^${symbol}\\\\]*${p}*${symbol}`;
}

const _allow_whitespace = "\\s*";
const _single_quoted_string = quotedStringPattern("'");
const _double_quoted_string = quotedStringPattern('"');
const _nonComma_nonQuote_stuff = `[^,'"\\s\\\\]*(?:\\s+[^,'"\\s\\\\]+)*`;

const _single_or_double_quoted_stuff = group(
  `${_single_quoted_string}|${_double_quoted_string}|${_nonComma_nonQuote_stuff}`
);
const _values_separated_by_comma = group(
  `'${_allow_whitespace}${_single_or_double_quoted_stuff}${_allow_whitespace}`
);

const re_valid = new RegExp(
  `^${_allow_whitespace}${_single_or_double_quoted_stuff}${_allow_whitespace}${_values_separated_by_comma}*$`
);

const _dont_match_empty_last_value = "(?!s*$)";
const _field_ends_on_comma_or_EOS = group(`,|$`);

const re_value = new RegExp(
  `${_dont_match_empty_last_value}${_allow_whitespace}${_single_or_double_quoted_stuff}${_allow_whitespace}${_field_ends_on_comma_or_EOS}`
);

export function csvToArray(text: string) {
  if (!re_valid.test(text)) {
    return [];
  }
  var a = [];
  text.replace(
    re_value, // "Walk" the string using replace with callback.
    function (m0, m1, m2, m3) {
      // Remove backslash from \' in single quoted values.
      if (m1 !== undefined) a.push(m1.replace(/\\'/g, "'"));
      // Remove backslash from \" in double quoted values.
      else if (m2 !== undefined) a.push(m2.replace(/\\"/g, '"'));
      else if (m3 !== undefined) a.push(m3);
      return ""; // Return empty string.
    }
  );
  // Handle special case of empty last value.
  if (/,\s*$/.test(text)) a.push("");
  return a;
}
