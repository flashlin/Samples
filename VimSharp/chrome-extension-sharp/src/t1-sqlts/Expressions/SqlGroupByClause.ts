import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression } from './ISqlExpression';

export class SqlGroupByClause implements ISqlExpression {
    SqlType: SqlType = SqlType.GroupByClause;
    Span: TextSpan = new TextSpan();
    Columns: ISqlExpression[] = [];

    Accept(visitor: any): void {
        // 簡單實作
    }

    ToSql(): string {
        return `GROUP BY ${this.Columns.map(c => c.ToSql()).join(', ')}`;
    }
} 