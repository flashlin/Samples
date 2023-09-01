import { parseLinq } from "./linq";

export function linqToSqlite(linqText: string) {
    const rc = parseLinq(linqText);
    console.log("parse", rc.value);

    let sql = "SELECT ";
    sql += rc.value.columns.map(column => {
        if( column.type == 'TABLE_FIELD' ) {
            return `${column.aliasTableName}.${column.field} AS ${column.aliasFieldName}`;
        }
    }).join(',');
    // return {
    //     value: value,
    //     lexResult: lexResult,
    //     parseErrors: parser.errors,
    // };
    return sql;
}
