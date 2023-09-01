import { parseLinq, type ITableFieldExpression, type ITableExpression } from "./linq";

export function linqToSqlite(linqText: string) {
    const rc = parseLinq(linqText);
    const expr = rc.value;

    let sql = "SELECT ";
    sql += expr.columns.map(column => {
        if( column.type == 'TABLE_FIELD' ) {
            const col = column as unknown as ITableFieldExpression;
            return `${column.aliasTableName}.${col.field} AS ${col.aliasFieldName}`;
        }
    }).join(',');

    sql += ' FROM ';
    if( expr.source.type == 'TABLE_CLAUSE' ) {
        const table = expr.source as unknown as ITableExpression;
        sql += `${table.name} AS ${expr.aliasTableName}`;
    }
    expr.aliasTableName;
    // return {
    //     value: value,
    //     lexResult: lexResult,
    //     parseErrors: parser.errors,
    // };
    return sql;
}
