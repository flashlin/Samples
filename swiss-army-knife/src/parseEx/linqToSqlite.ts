import { parseLinq } from "./linq";

export function linqToSqlite(linqText: string) {
    const rc = parseLinq(linqText);
    // return {
    //     value: value,
    //     lexResult: lexResult,
    //     parseErrors: parser.errors,
    // };
    return rc.value;
}
