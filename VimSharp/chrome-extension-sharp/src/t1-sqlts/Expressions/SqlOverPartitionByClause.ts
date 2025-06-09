import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { SqlOrderColumn } from './SqlOrderColumn';
import { SqlFieldExpr } from './SqlFieldExpr';

export class SqlOverPartitionByClause implements ISqlExpression {
    SqlType: SqlType = SqlType.OverPartitionByClause;
    Span: TextSpan = new TextSpan();
    Field: ISqlExpression = new SqlFieldExpr();
    By: ISqlExpression[] = [];
    Columns: SqlOrderColumn[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_OverPartitionByClause(this);
    }

    ToSql(): string {
        let sql = this.Field.ToSql() + ' OVER (';
        sql += 'PARTITION BY ' + this.By.map(x => x.ToSql()).join(',');
        sql += 'ORDER BY ' + this.Columns.map(x => x.ToSql()).join(', ');
        sql += ')';
        return sql;
    }
} 