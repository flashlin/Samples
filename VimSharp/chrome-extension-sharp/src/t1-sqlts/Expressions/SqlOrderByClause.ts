import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { SqlOrderColumn } from './SqlOrderColumn';

export class SqlOrderByClause implements ISqlExpression {
    SqlType: SqlType = SqlType.OrderByClause;
    Span: TextSpan = new TextSpan();
    Columns: SqlOrderColumn[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_OrderByClause(this);
    }

    ToSql(): string {
        let sql = 'ORDER BY\n';
        sql += this.Columns.map(x => x.ToSql()).join(',\n');
        return sql;
    }
} 