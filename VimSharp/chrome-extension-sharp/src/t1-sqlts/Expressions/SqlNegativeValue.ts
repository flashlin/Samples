import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlNegativeValue implements ISqlExpression {
    SqlType: SqlType = SqlType.NegativeValue;
    Span: TextSpan = new TextSpan();
    Value!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_NegativeValue(this);
    }

    ToSql(): string {
        return `-${this.Value.ToSql()}`;
    }
} 