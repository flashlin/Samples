import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlPartitionBy implements ISqlExpression {
    SqlType: SqlType = SqlType.PartitionBy;
    Span: TextSpan = new TextSpan();
    Columns: ISqlExpression[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_PartitionBy(this);
    }

    ToSql(): string {
        let sql = 'PARTITION BY ';
        sql += this.Columns.map(x => x.ToSql()).join(', ');
        return sql;
    }
} 