import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlNotExpression implements ISqlExpression {
    SqlType: SqlType = SqlType.NotExpression;
    Span: TextSpan = new TextSpan();
    Value!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_NotExpression(this);
    }

    ToSql(): string {
        return `NOT ${this.Value.ToSql()}`;
    }
} 