import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { SqlPartitionBy } from './SqlPartitionBy';
import { SqlOrderByClause } from './SqlOrderByClause';

export class SqlRankClause implements ISqlExpression {
    SqlType: SqlType = SqlType.RankClause;
    Span: TextSpan = new TextSpan();
    PartitionBy?: SqlPartitionBy;
    OrderBy!: SqlOrderByClause;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_RankClause(this);
    }

    ToSql(): string {
        let sql = '';
        if (this.PartitionBy) {
            sql += this.PartitionBy.ToSql() + ' ';
        }
        sql += this.OrderBy.ToSql();
        return sql;
    }
} 