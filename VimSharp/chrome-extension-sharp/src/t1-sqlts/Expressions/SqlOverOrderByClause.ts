import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { SqlOrderColumn } from './SqlOrderColumn';
import { SqlFieldExpr } from './SqlFieldExpr';

export class SqlOverOrderByClause implements ISqlExpression {
    SqlType: SqlType = SqlType.OverOrderBy;
    Span: TextSpan = new TextSpan();
    Field: ISqlExpression = new SqlFieldExpr();
    Columns: SqlOrderColumn[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_OverOrderByClause(this);
    }

    ToSql(): string {
        let sql = this.Field.ToSql() + ' OVER (';
        if (this.Columns.length > 0) {
            sql += 'ORDER BY ' + this.Columns.map((col) => col.ToSql()).join(', ');
        }
        sql += ')';
        return sql;
    }
} 