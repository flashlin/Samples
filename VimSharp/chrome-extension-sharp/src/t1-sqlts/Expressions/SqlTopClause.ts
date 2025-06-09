import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlTopClause implements ISqlExpression {
    SqlType: SqlType = SqlType.TopClause;
    Span: TextSpan = new TextSpan();
    Expression!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_TopClause(this);
    }

    ToSql(): string {
        return `TOP ${this.Expression.ToSql()}`;
    }
} 