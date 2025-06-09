import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlGroupByClause implements ISqlExpression {
    SqlType: SqlType = SqlType.GroupByClause;
    Span: TextSpan = new TextSpan();
    Columns: ISqlExpression[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_GroupByClause(this);
    }

    ToSql(): string {
        return 'GROUP BY ' + this.Columns.map(x => x.ToSql()).join(', ');
    }
} 