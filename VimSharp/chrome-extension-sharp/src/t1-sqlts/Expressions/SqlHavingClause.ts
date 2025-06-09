import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { SqlValue } from './SqlValue';

export class SqlHavingClause implements ISqlExpression {
    SqlType: SqlType = SqlType.HavingClause;
    Span: TextSpan = new TextSpan();
    Condition: ISqlExpression = new SqlValue();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_HavingClause(this);
    }

    ToSql(): string {
        return 'HAVING ' + this.Condition.ToSql();
    }
} 